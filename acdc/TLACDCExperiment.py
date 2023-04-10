import pickle
import gc
from typing import Callable, Optional, Literal, List, Dict, Any, Tuple, Union, Set, Iterable, TypeVar, Type
import random
from dataclasses import dataclass
import torch
from acdc.graphics import show
from torch import nn
from torch.nn import functional as F
from acdc.TLACDCInterpNode import TLACDCInterpNode
from acdc.TLACDCCorrespondence import TLACDCCorrespondence, TLACDCCorrespondenceFast
from acdc.HookedTransformer import HookedTransformer
from acdc.graphics import log_metrics_to_wandb
import warnings
import wandb
from acdc.acdc_utils import TorchIndex, Edge, EdgeType
from collections import OrderedDict
from functools import partial
import time

def next_key(ordered_dict: OrderedDict, current_key):
    key_iterator = iter(ordered_dict)
    next((key for key in key_iterator if key == current_key), None)
    return next(key_iterator, None)

class TLACDCExperiment:
    """Manages an ACDC experiment, including the computational graph, the model, the data etc.
    Based off of ACDCExperiment from rust_circuit code"""

    def __init__(
        self,
        model: HookedTransformer,
        ds: torch.Tensor,
        ref_ds: Optional[torch.Tensor],
        threshold: float,
        metric: Callable[[torch.Tensor, torch.Tensor], float], # dataset and logits to metric
        second_metric: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None,
        verbose: bool = False,
        hook_verbose: bool = False,
        parallel_hypotheses: int = 1, # lol
        remove_redundant: bool = False, # TODO implement
        monotone_metric: Literal[
            "off", "maximize", "minimize"
        ] = "minimize",  # if this is set to "maximize" or "minimize", then the metric will be maximized or minimized, respectively instead of us trying to keep the metric roughly the same. We do KL divergence by default
        first_cache_cpu: bool = True,
        second_cache_cpu: bool = True,
        zero_ablation: bool = False, # use zero rather than 
        using_wandb: bool = False,
        wandb_entity_name: str = "",
        wandb_project_name: str = "",
        wandb_run_name: str = "",
        wandb_notes: str = "",
        skip_edges = "yes",
        add_sender_hooks: bool = True,
        add_receiver_hooks: bool = False,
        indices_mode: Literal["normal", "reverse", "shuffle"] = "reverse", # we get best performance with reverse I think
        names_mode: Literal["normal", "reverse", "shuffle"] = "normal",
        fast_mode=False,
    ):
        """Initialize the ACDC experiment"""

        self.indices_mode = indices_mode
        self.names_mode = names_mode

        self.model = model
        self.verify_model_setup()
        self.zero_ablation = zero_ablation
        self.verbose = verbose
        self.step_idx = 0
        self.fast_mode = fast_mode
        self.hook_verbose = hook_verbose
        self.skip_edges = skip_edges
        if skip_edges != "yes":
            raise NotImplementedError() # TODO if edge counts are slow...

        self.corr = TLACDCCorrespondence.setup_from_model(self.model)
            
        self.reverse_topologically_sort_corr()
        self.current_node = self.corr.first_node()
        print(f"{self.current_node=}")

        if self.fast_mode:
            raise NotImplementedError("Need to turn other things into template_corr...")
            self.corr = TLACDCCorrespondenceFast()
            self.add_node(self.current_node) # start with just the output node
        else:
            self.corr = self.corr

        self.ds = ds
        self.ref_ds = ref_ds
        self.first_cache_cpu = first_cache_cpu
        self.second_cache_cpu = second_cache_cpu
        self.setup_second_cache()
        if self.second_cache_cpu:
            self.model.global_cache.to("cpu", which_caches="second")
        self.setup_model_hooks(
            add_sender_hooks=add_sender_hooks,
            add_receiver_hooks=add_receiver_hooks,
        )

        self.using_wandb = using_wandb
        if using_wandb:
            wandb.init(
                entity=wandb_entity_name,
                project=wandb_project_name,
                name=wandb_run_name,
                notes=wandb_notes,
            )

        self.metric = metric
        self.second_metric = second_metric
        self.update_cur_metric()

        self.threshold = threshold
        assert self.ref_ds is not None or self.zero_ablation, "If you're doing random ablation, you need a ref ds"

        self.parallel_hypotheses = parallel_hypotheses
        if self.parallel_hypotheses != 1:
            raise NotImplementedError("Parallel hypotheses not implemented yet") # TODO?

        if self.using_wandb:
            # TODO?
            self.metrics_to_plot = {}
            self.metrics_to_plot["new_metrics"] = []
            self.metrics_to_plot["list_of_parents_evaluated"] = []
            self.metrics_to_plot["list_of_children_evaluated"] = []
            self.metrics_to_plot["list_of_nodes_evaluated"] = []
            self.metrics_to_plot["evaluated_metrics"] = []
            self.metrics_to_plot["current_metrics"] = []
            self.metrics_to_plot["results"] = []
            self.metrics_to_plot["acdc_step"] = 0
            self.metrics_to_plot["num_edges"] = []
            self.metrics_to_plot["times"] = []
            self.metrics_to_plot["times_diff"] = []

    def verify_model_setup(self):
        assert self.model.cfg.use_attn_result, "Need to be able to see split by head outputs"
        assert self.model.cfg.use_split_qkv_input, "Need to be able to see split by head QKV inputs"
        assert self.model.cfg.use_global_cache, "Need to be able to use global chache to do ACDC"


    def add_node(self, node: TLACDCInterpNode): # TODO rename "receiver" node ???
        """Add a node to our current ACDC Hypothesis"""

        self.corr.add_node(node, safe=True)

        self.model.add_hook(
            name=node.name,
            hook=partial(self.receiver_hook, verbose=self.hook_verbose),
        )

        # add all the god-damn edges too?
        for sender_name in self.corr[node.name][node.index]:
            for sender_idx in self.corr[node.name][node.index][sender_name]:
                self.corr.edges[node.name][node.index][sender_name][sender_idx] = self.corr.edges[node.name][node.index][sender_name][sender_idx]
                assert False

    def update_cur_metric(self, recalc=True, initial=False):
        if recalc:
            logits = self.model(self.ds)
            self.cur_metric = self.metric(logits)
            if self.second_metric is not None:
                self.cur_second_metric = self.second_metric(logits)

        if initial:
            assert abs(self.cur_metric) < 1e-5, f"Metric {self.cur_metric=} is not zero"

        if self.using_wandb:
            wandb_return_dict = {
                "cur_metric": self.cur_metric,
                "num_edges": self.count_no_edges(),
            }
            if self.second_metric is not None:
                wandb_return_dict["second_cur_metric"] = self.cur_second_metric
                wandb.log(wandb_return_dict)

    def reverse_topologically_sort_corr(self):
        """Topologically sort the template corr"""
        for hook in self.model.hook_dict.values():
            assert len(hook.fwd_hooks) == 0, "Don't load the model with hooks *then* call this"

        new_graph = OrderedDict()
        cache=OrderedDict() # what if?
        self.model.cache_all(cache)
        self.model(torch.arange(5)) # some random forward pass so that we can see all the hook names
        self.model.reset_hooks()

        if self.verbose:
            print(self.corr.graph.keys())

        cache_keys = list(cache.keys())
        cache_keys.reverse()

        for hook_name in cache_keys:
            print(hook_name)            
            if hook_name in self.corr.graph:
                new_graph[hook_name] = self.corr.graph[hook_name]

        self.corr.graph = new_graph

    def sender_hook(self, z, hook, verbose=False, cache="first", device=None):
        """General, to cover online and corrupt caching"""

        if device == "cpu":
            tens = z.cpu()
        else:
            tens = z.clone()
            if device is not None:
                tens = tens.to(device)

        if cache == "second":
            hook.global_cache.second_cache[hook.name] = tens
        elif cache == "first":
            hook.global_cache.cache[hook.name] = tens
        else:
            raise ValueError(f"Unknown cache type {cache}")

        if verbose:
            print(f"Saved {hook.name} with norm {z.norm().item()}")

        return z

    def receiver_hook(self, z, hook, verbose=False):
        receiver_node_name = hook.name

        z[:] = self.model.global_cache.second_cache[hook.name].to(z.device) # TODO - is this slow ???

        for receiver_node_index in self.corr.edges[hook.name]:
            direct_computation_nodes = []
            for sender_node_name in self.corr.edges[hook.name][receiver_node_index]:
                for sender_node_index in self.corr.edges[hook.name][receiver_node_index][sender_node_name]:

                    edge = self.corr.edges[hook.name][receiver_node_index][sender_node_name][sender_node_index] # TODO maybe less crazy nested indexes ... just make local variables each time?

                    if not edge.present:
                        continue # don't do patching stuff, if it wastes time

                    if verbose:
                        print(
                            hook.name, receiver_node_index, sender_node_name, sender_node_index,
                        )
                        print("-------")
                        if edge.edge_type == EdgeType.ADDITION:
                            print(
                                hook.global_cache.cache[sender_node_name].shape,
                                sender_node_index,
                            )
                    
                    if edge.edge_type == EdgeType.ADDITION:
                        z[receiver_node_index.as_index] += hook.global_cache.cache[
                            sender_node_name
                        ][sender_node_index.as_index].to(z.device)
                        z[receiver_node_index.as_index] -= hook.global_cache.second_cache[
                            sender_node_name
                        ][sender_node_index.as_index].to(z.device)

                    elif edge.edge_type == EdgeType.DIRECT_COMPUTATION:
                        direct_computation_nodes.append(self.corr.graph[sender_node_name][sender_node_index])
                        assert len(direct_computation_nodes) == 1, f"Found multiple direct computation nodes {direct_computation_nodes}"

                        z[receiver_node_index.as_index] = hook.global_cache.second_cache[receiver_node_name][receiver_node_index.as_index].to(z.device)

                    else: 
                        print(edge)
                        raise ValueError(f"Unknown edge type {edge.edge_type}")

        return z

    def add_all_sender_hooks(self, reset=True, cache="first", skip_direct_computation=False, add_all_hooks=False):
        """We use add_sender_hook for lazily adding *some* sender hooks"""

        if self.verbose:
            print("Adding sender hooks...")
        if reset:
            self.model.reset_hooks()
        device = {
            "first": "cpu" if self.first_cache_cpu else None,
            "second": "cpu" if self.second_cache_cpu else None,
        }[cache]

        nodes = []

        for big_tuple, edge in self.corr.all_edges().items():
            if edge.edge_type == EdgeType.DIRECT_COMPUTATION:
                if not skip_direct_computation:
                    nodes.append(self.corr.graph[big_tuple[0]][big_tuple[1]])
                    if add_all_hooks:
                        nodes.append(self.corr.graph[big_tuple[2]][big_tuple[3]])
            elif edge.edge_type == EdgeType.ADDITION:
                nodes.append(self.corr.graph[big_tuple[2]][big_tuple[3]])
                if add_all_hooks:
                    nodes.append(self.corr.graph[big_tuple[0]][big_tuple[1]])
            elif edge.edge_type != EdgeType.PLACEHOLDER:
                print(edge.edge_type.value, EdgeType.ADDITION.value, edge.edge_type.value == EdgeType.ADDITION.value, type(edge.edge_type.value), type(EdgeType.ADDITION.value))
                raise ValueError(f"{str(big_tuple)} {str(edge)} failed")
            print(big_tuple, "worked!")

            for node in nodes:
                if len(self.model.hook_dict[node.name].fwd_hooks) > 0:
                    for hook_func_maybe_partial in self.model.hook_dict[node.name].fwd_hook_functions:
                        hook_func_name = hook_func_maybe_partial.func.__name__ if isinstance(hook_func_maybe_partial, partial) else hook_func_maybe_partial.__name__
                        assert "sender_hook" in hook_func_name, f"You should only add sender hooks to {node.name}, and this: {hook_func_name} doesn't look like a sender hook"
                    continue

                self.model.add_hook(
                    name=node.name, 
                    hook=partial(self.sender_hook, verbose=self.hook_verbose, cache=cache, device=device),
                )

    def setup_second_cache(self):
        if self.verbose:
            print("Adding sender hooks...")
        self.add_all_sender_hooks(cache="second", add_all_hooks=True)

        if self.verbose:
            print("Now corrupting things..")

        corrupt_stuff = self.model(self.ref_ds)

        if self.verbose:
            print("Done corrupting things")

        if self.zero_ablation:
            names = list(self.model.global_cache.second_cache.keys())
            assert len(names)>0, "No second cache names found"
            print("WE NAAMING")
            for name in names:
                self.model.global_cache.second_cache[name] = torch.zeros_like(
                    self.model.global_cache.second_cache[name]
                )
                torch.cuda.empty_cache()

        if self.second_cache_cpu:
            self.model.global_cache.to("cpu", which_caches="second")

        self.model.reset_hooks()

    def setup_model_hooks(
        self, 
        add_sender_hooks=False,
        add_receiver_hooks=False,
    ):
        if add_sender_hooks:
            self.add_all_sender_hooks(cache="first", skip_direct_computation=True) # remove because efficiency 

        if add_receiver_hooks:
            
            warnings.warn("Deprecating adding receiver hooks before launching into ACDC runs, this may be totally broke")

            receiver_node_names = list(set([node.name for node in self.corr.nodes()]))
            for receiver_name in receiver_node_names: # TODO could remove the nodes that don't have any parents...
                self.model.add_hook(
                    name=receiver_name,
                    hook=partial(self.receiver_hook, verbose=self.hook_verbose),
                )

    def save_edges(self, fname):
        """Stefan's idea for fast saving!
        TODO pickling of the whole experiment work"""

        edges_list = []
        for t, e in self.corr.all_edges().items():
            if e.present and e.edge_type != EdgeType.PLACEHOLDER:
                edges_list.append((t, e.effect_size))
        
        with open(fname, "wb") as f:
            pickle.dump(edges_list, f)

    def add_sender_hook(self, node):
        if len(self.model.hook_dict[node.name].fwd_hooks) > 0:
            for hook_func_maybe_partial in self.model.hook_dict[node.name].fwd_hook_functions:
                hook_func_name = hook_func_maybe_partial.func.__name__ if isinstance(hook_func_maybe_partial, partial) else hook_func_maybe_partial.__name__
                assert "sender_hook" in hook_func_name, f"You should only add sender hooks to {node.name}, and this: {hook_func_name} doesn't look like a sender hook"
            return False # already added, whatever move on

        handle = self.model.add_hook(
            name=node.name, 
            hook=partial(self.sender_hook, verbose=self.hook_verbose, cache="first", device="cpu" if self.first_cache_cpu else None),
        )

        return True

    def add_receiver_hook(self, node):
        if len(self.model.hook_dict[node.name].fwd_hooks) > 0: # repeating code from add_sender_hooks
            for hook_func_maybe_partial in self.model.hook_dict[node.name].fwd_hook_functions:
                hook_func_name = hook_func_maybe_partial.func.__name__ if isinstance(hook_func_maybe_partial, partial) else hook_func_maybe_partial.__name__
                assert "receiver_hook" in hook_func_name, f"You should only add receiver hooks to {node.name}, and this: {hook_func_name} doesn't look like a receiver hook"
            return False # already added, whatever move on

        handle = self.model.add_hook(
            name=node.name,
            hook=partial(self.receiver_hook, verbose=self.hook_verbose),
        )

        return True


    def step(self, early_stop=False):
        if self.current_node is None:
            return

        start_step_time = time.time()
        self.step_idx += 1

        self.update_cur_metric()
        initial_metric = self.cur_metric
        assert isinstance(initial_metric, float), f"Initial metric is a {type(initial_metric)} not a float"

        cur_metric = initial_metric
        if self.verbose:
            print("New metric:", cur_metric)

        if self.current_node.incoming_edge_type.value != EdgeType.PLACEHOLDER.value:
            # Add this node as a receiver hook, now
            added_receiver_hook = self.add_receiver_hook(self.current_node)
            if added_receiver_hook:
                added_receiver_hook = self.model.hook_dict[self.current_node.name].fwd_hooks[-1]
        else:
            added_receiver_hook = False

        is_this_node_used = False
        sender_names_list = list(self.corr.edges[self.current_node.name][self.current_node.index])

        if self.names_mode == "random":
            random.shuffle(sender_names_list)
        elif self.names_mode == "reverse":
            sender_names_list = list(reversed(sender_names_list))

        for sender_name in sender_names_list:
            sender_indices_list = list(self.corr.edges[self.current_node.name][self.current_node.index][sender_name])

            if self.indices_mode == "random":
                random.shuffle(sender_indices_list)
            elif self.indices_mode == "reverse":
                sender_indices_list = list(reversed(sender_indices_list))

            for sender_index in sender_indices_list:
                edge = self.corr.edges[self.current_node.name][self.current_node.index][sender_name][sender_index]
                cur_parent = self.corr.graph[sender_name][sender_index]

                if edge.edge_type == EdgeType.PLACEHOLDER:
                    is_this_node_used = True
                    continue # include by default

                if self.verbose:
                    print(f"\nNode: {cur_parent=} ({self.current_node=})\n")

                edge.present = False

                if edge.edge_type == EdgeType.ADDITION:
                    added_sender_hook = self.add_sender_hook(
                        cur_parent,
                    )
                else:
                    added_sender_hook = False
                
                old_metric = self.cur_metric
                self.update_cur_metric()
                evaluated_metric = self.cur_metric # self.metric(self.model(self.ds)) # OK, don't calculate second metric?

                if early_stop: # for debugging the effects of one and only one forward pass WITH a corrupted edge
                    return

                if self.verbose:
                    print(
                        "Metric after removing connection to",
                        sender_name,
                        sender_index,
                        "is",
                        evaluated_metric,
                        "(and current metric " + str(cur_metric) + ")",
                    )

                result = evaluated_metric - old_metric
                edge.effect_size = result

                if self.verbose:
                    print("Result is", result, end="")

                if result < self.threshold:
                    print("...so removing connection")
                    self.cur_metric = evaluated_metric

                else: # include this edge in the graph
                    is_this_node_used = True

                    if self.verbose:
                        print("...so keeping connection")
                    edge.present = True

                    if added_sender_hook:
                        self.model.hook_dict[sender_name].remove_hooks()

                    self.update_cur_metric(recalc = False) # so we log current state to wandb

                if self.using_wandb:
                    log_metrics_to_wandb(
                        self,
                        current_metric = cur_metric,
                        parent_name = str(self.corr.graph[sender_name][sender_index]),
                        child_name = str(self.current_node),
                        result = result,
                        times = time.time(),
                    )

            self.update_cur_metric()

        # TODO find an efficient way to do this...
        # if added_receiver_hook and not is_this_node_used:
        #     assert self.model.hook_dict[self.current_node.name].fwd_hooks[-1] == added_receiver_hook, f"You should not have added additional hooks to {self.current_node.name}..."
        #     added_receiver_hook = self.model.hook_dict[self.current_node.name].fwd_hooks.pop()
        #     added_receiver_hook.hook.remove()

        if is_this_node_used and self.current_node.incoming_edge_type.value != EdgeType.PLACEHOLDER.value:
            fname = f"ims/img_new_{self.step_idx}.png"
            show(
                self.corr,
                fname=fname,
                show_full_index=False, # hopefully works
            )
            if self.using_wandb:
                wandb.log(
                    {"acdc_graph": wandb.Image(fname),}
                )

        # increment the current node
        self.increment_current_node()
        self.update_cur_metric(recalc=False)

    def current_node_connected(self):
        for child_name, rest1 in self.corr.edges.items(): # rest1 just meaning "rest of dictionary.. I'm tired"
            for child_index, rest2 in rest1.items():
                if self.current_node.name in rest2 and self.current_node.index in rest2[self.current_node.name]:
                    if rest2[self.current_node.name][self.current_node.index].present:
                        return True

        # if this is NOT connected, then remove all incoming edges, too

        self.update_cur_metric()
        old_metric = self.cur_metric

        for parent_name, rest1 in self.corr.edges[self.current_node.name][self.current_node.index].items():
            for parent_index, rest2 in rest1.items():
                edge = self.corr.edges[self.current_node.name][self.current_node.index][parent_name][parent_index]
                edge.present=False

                self.update_cur_metric()
                assert abs(self.cur_metric - old_metric) < 1e-3, ("Removing all incoming edges should not change the metric", self.cur_metric, old_metric, self.current_node, parent_name, parent_index)

        return False

    def find_next_node(self) -> Optional[TLACDCInterpNode]:
        next_index = next_key(self.corr.graph[self.current_node.name], self.current_node.index)
        if next_index is not None:
            return self.corr.graph[self.current_node.name][next_index]

        next_name = next_key(self.corr.graph, self.current_node.name)

        if next_name is not None:
            return list(self.corr.graph[next_name].values())[0]
            
        warnings.warn("Finished iterating")
        return None

    def increment_current_node(self) -> None:
        self.current_node = self.find_next_node()
        print("We moved to ", self.current_node)

        if self.current_node is not None and not self.current_node_connected():
            print("But it's bad")
            self.increment_current_node()

        if self.fast_mode:
            self.add_node(self.current_node, safe=True)

    def count_no_edges(self):
        cnt = 0

        for edge in self.corr.all_edges().values():
            if edge.present and edge.edge_type != EdgeType.PLACEHOLDER:
                cnt += 1

        if self.verbose:
            print("No edge", cnt)
        return cnt