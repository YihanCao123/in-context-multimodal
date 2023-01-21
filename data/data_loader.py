import torchvision.datasets as dset
import torchvision.transforms as transforms

from transformers import BertTokenizerFast

import torch
from torch.utils.data import Dataset, DataLoader

class CocoDataset(Dataset):
    
    def __init__(self, img_root, ann_path, transform=None, args=None):
        self.p = args
        self.dataset = dset.CocoCaptions(root=img_root,
                                         annFile=ann_path,
                                         transform=transform)
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.tokenizer.add_special_tokens(["[IMG]"])
        self.vocab = self.tokenizer.get_vocab()
    
    def __getitem__(self, index):
        image, captions = self.dataset[index]
        caption = "[CLS] " + captions[0] + " [SEP]"
        return image, caption
    
    def __len__(self):
        return len(self.dataset)

    def make_prompt(self, cap):
        # make single prompt
        return

    def make_in_context_prompt(self, cap, n_shots):
        # make in context n_shots prompt
        return
    
    def pad_data(self, caps):
        encoding = self.tokenizer(caps, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])
        token_type_ids = torch.LongTensor(encoding['token_type_ids'])

        return token_ids, token_type_ids, attention_mask, caps
    
    def collate_fn(self, all_data):
        image, caption = all_data
        token_ids, token_type_ids, attention_mask, caption = self.pad_data(caption)
        return {
            'img': image,
            'cap': caption,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
        }


if __name__ == "__main__":
    train_dataset = CocoDataset("val2017", "annotations/captions_val2017.json", transforms.PILToTensor())

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8, collate_fn=train_dataset.collate_fn)

    
    

                                        