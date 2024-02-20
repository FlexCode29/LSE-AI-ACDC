from functools import partial
import torch
from transformer_lens.HookedTransformer import HookedTransformer
from acdc.docstring.utils import AllDataThings
import torch.nn.functional as F

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
    return data_tokens, correct_answers_tensor





def get_all_icl_things(device='cpu', x=1, y=1, icl_length=12, n=10, return_one_element: bool = False) -> AllDataThings:
    model = HookedTransformer.from_pretrained("gpt2").to(device)

    # Generate validation data
    validation_data, validation_correct_answers = generate_data(model, device, x, y, icl_length, n)

    # Generate validation patch data with different x and y values (using offset)
    validation_patch_data, _ = generate_data(model, device, x, y, icl_length, n // 2, offset=10)

    # Generate separate test data with further different x and y values
    test_data, test_correct_answers = generate_data(model, device, x, y, icl_length, n // 2, offset=20)

    # Generate test patch data with different x and y values (using offset)
    test_patch_data, _ = generate_data(model, device, x, y, icl_length, n // 2, offset=30)


    def validation_metric(model_output, correct):
        # Decode the model's predictions
        top_tokens = model_output.topk(1).indices.squeeze(-1)
        predictions = model.to_string(top_tokens)
        predictions = torch.tensor([int(pred) for pred in predictions]).to(torch.double).unsqueeze(-1).to(device)

        # Calculate MSE
        mse = F.mse_loss(predictions, correct, reduction='mean' if not return_one_element else 'sum')
        return mse

    return AllDataThings(
        tl_model=model,
        validation_metric=partial(validation_metric, correct=validation_correct_answers),
        validation_data=validation_data,
        validation_labels=validation_correct_answers,
        validation_mask=None,
        validation_patch_data=validation_patch_data,
        test_metrics=partial(validation_metric, correct=test_correct_answers),
        test_data=test_data,
        test_labels=test_correct_answers,
        test_mask=None,
        test_patch_data=test_patch_data,
    )


# Testing:
import unittest
class TestICLDataAndModel(unittest.TestCase):
    def setUp(self):
        self.device = 'cpu'
        self.model = HookedTransformer.from_pretrained("gpt2").to(self.device)
        self.x_initial = 1
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
        _, correct_answers = generate_data(self.model, self.device, self.x_initial, self.y_initial, self.icl_length, self.n)
        mock_model_output = torch.randn(size=(correct_answers.shape[0], 1), device=self.device)
        # Assuming validation_metric is defined elsewhere in your code
        mse = validation_metric(mock_model_output, correct_answers)
        self.assertIsInstance(mse, torch.Tensor)

# Execute tests when the script is run
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)