from functools import partial
import torch
from transformer_lens.HookedTransformer import HookedTransformer
from acdc.docstring.utils import AllDataThings
import torch.nn.functional as F
from numpy import array


def get_gpt2(device="cpu"):
    tl_model = HookedTransformer.from_pretrained("gpt2")
    tl_model = tl_model.to(device)
    tl_model.set_use_attn_result(True)
    tl_model.set_use_split_qkv_input(True)
    if "use_hook_mlp_in" in tl_model.cfg.to_dict():
        tl_model.set_use_hook_mlp_in(True)
    return tl_model

def generate_data(model, device, x_initial, y_initial, icl_length, n, offset=0):
    prompts = []
    correct_answers = []
    # Initialize x and y for the sequence
    x, y = x_initial + offset, y_initial + offset


    for i in range(n):
        prompt = ''
        for j in range(icl_length + 1):
            if j < icl_length:
                prompt += f"Input: {j}, Output: {j * x + y}\n"
            else:
                prompt += f"Input: {j}, Output:"
                correct_answers.append(j * x + y)  # Record the correct answer for the last input
        prompts.append(prompt)
        # Update x and y after generating each full prompt set
        if i % 2 == 0:
            x += 1
        else:
            y += 1


    # Convert prompts into tokens
    data_tokens = [model.to_tokens(prompt).to(device) for prompt in prompts]
    correct_answers_tensor = torch.tensor(correct_answers).to(torch.double).unsqueeze(-1).to(device)
    return prompts, correct_answers_tensor




def get_all_icl_things(device='cpu', x=2, y=1, icl_length=12, n=10, return_one_element: bool = False) -> AllDataThings:
    model = get_gpt2(device)

    # Generate validation data
    validation_data, validation_correct_answers = generate_data(model, device, x, y, icl_length, n, offset=0)

    # Generate validation patch data with different x and y values (using offset)
    validation_patch_data, _ = generate_data(model, device, x, y, icl_length, n, offset=n)

    # Generate separate test data with further different x and y values
    test_data, test_correct_answers = generate_data(model, device, x, y, icl_length, n, offset= n * 2)

    # Generate test patch data with different x and y values (using offset)
    test_patch_data, _ = generate_data(model, device, x, y, icl_length, n, offset= n * 3)


    def validation_metric(output, correct, return_one_element, device):

        output = output.to(device)
    
        # Select the logits for the last token in each sequence
        # model_output shape: [batch_size, seq_length, vocab_size] => [10, 103, 50257]
        # We select [:, -1, :] to get the last token logits for each example in the batch
        last_token_logits = output[:, -1, :]  # Shape: [10, 50257]
    
        # Now, find the indices of the 10 highest logits for the last token across the batch
        # We use torch.topk to get the top 10 logits' indices for each example
        topk_values, topk_indices = torch.topk(last_token_logits, 1, dim=1) 

        predictions = model.to_string(topk_indices)
        predictions = torch.tensor([int(pred) for pred in predictions]).to(torch.double).unsqueeze(-1).to(device)

        # Calculate MSE
        mse = F.mse_loss(predictions, correct, reduction='mean' if not return_one_element else 'sum')
        return mse

    return AllDataThings(
        tl_model=model,
        validation_metric=partial(validation_metric, correct=validation_correct_answers, return_one_element=return_one_element, device=device),
        validation_data=validation_data,
        validation_labels=validation_correct_answers,
        validation_mask=None,
        validation_patch_data=validation_patch_data,
        test_metrics=partial(validation_metric, correct=test_correct_answers, return_one_element=return_one_element, device=device),
        test_data=test_data,
        test_labels=test_correct_answers,
        test_mask=None,
        test_patch_data=test_patch_data,
    )


# Testing (pay attention, this is now buggd and would require valid function outside of get things):
'''
import unittest
class TestICLDataAndModel(unittest.TestCase):
    def setUp(self):
        self.device = 'cpu'
        self.model = HookedTransformer.from_pretrained("gpt2").to(self.device)
        self.x_initial = 2
        self.y_initial = 1
        self.icl_length = 12
        self.n = 10

    def test_data_generation(self):
        data, correct_answers = generate_data(self.model, self.device, self.x_initial, self.y_initial, self.icl_length, self.n)
        # Ensure data and correct answers are generated for each prompt
        self.assertEqual(len(data), self.n)
        self.assertTrue(correct_answers.shape[0], self.n)
        # Ensure the shape of correct_answers tensor is as expected
        self.assertEqual(correct_answers.shape, (self.n, 1))

    def test_validation_metric(self):
        data, correct_answers = generate_data(self.model, self.device, self.x_initial, self.y_initial, self.icl_length, self.n)
        # Assuming validation_metric is defined elsewhere in your code
        logits = self.model(data, return_type="logits")
        mse = validation_metric(model=self.model, model_output=logits, correct=correct_answers, return_one_element=False, device=self.device)
        self.assertIsInstance(mse, torch.Tensor)
        print('This is the MSE: ', mse)

# Execute tests when the script is run
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
'''