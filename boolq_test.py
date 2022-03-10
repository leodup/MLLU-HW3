import data_utils
import pandas as pd
import torch
import unittest

from boolq import BoolQDataset
from transformers import RobertaTokenizerFast


class TestBoolQDataset(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.dataset = pd.DataFrame.from_dict(
            {
                "question": ["question 0", "question 1"],
                "passage": ["passage 0", "passage 1"],
                "idx": [0, 1],
                "label": [True, False],
            }
        )
        self.max_seq_len = 4

        self.boolq_dataset = BoolQDataset(
            self.dataset, self.tokenizer, self.max_seq_len
        )

    def test_len(self):
        ## TODO: Test that the length of self.boolq_dataset is correct.
        ## len(self.boolq_dataset) should equal len(self.dataset).
        self.assertEqual(len(self.boolq_dataset), len(self.dataset))
        pass

    def test_item(self):
        ## TODO: Test that, for each element of self.boolq_dataset, 
        ## the output of __getitem__ (accessible via self.boolq_dataset[idx])
        ## has the correct keys, value dimensions, and value types.
        ## Each item should have keys ["input_ids", "attention_mask", "labels"].
        ## The input_ids and attention_mask values should both have length self.max_seq_len
        ## and type torch.long. The labels value should be a single numeric value.

        # Keys
        self.assertEqual(list(self.boolq_dataset.__getitem__(0).keys()), ["input_ids", "attention_mask", "labels"])
        self.assertEqual(list(self.boolq_dataset.__getitem__(1).keys()), ["input_ids", "attention_mask", "labels"])

        # input_ids dimensions & types
        self.assertEqual(len(self.boolq_dataset.__getitem__(0)['input_ids']), self.max_seq_len) # 0
        self.assertTrue(self.boolq_dataset.__getitem__(0)['input_ids'].dtype == torch.long)
        self.assertEqual(len(self.boolq_dataset.__getitem__(1)['input_ids']), self.max_seq_len) # 1
        self.assertTrue(self.boolq_dataset.__getitem__(1)['input_ids'].dtype == torch.long)
        
        # attention_mask dimensions & types
        self.assertEqual(len(self.boolq_dataset.__getitem__(0)['attention_mask']), self.max_seq_len) # 0
        self.assertTrue(self.boolq_dataset.__getitem__(0)['attention_mask'].dtype == torch.long)
        self.assertEqual(len(self.boolq_dataset.__getitem__(1)['attention_mask']), self.max_seq_len) # 1
        self.assertTrue(self.boolq_dataset.__getitem__(1)['attention_mask'].dtype == torch.long)

        # labels dimensions & types
        self.assertTrue(type(self.boolq_dataset.__getitem__(0)['labels']) == int) # 0
        self.assertTrue(type(self.boolq_dataset.__getitem__(1)['labels']) == int) # 1

        pass

if __name__ == "__main__":
    unittest.main()
