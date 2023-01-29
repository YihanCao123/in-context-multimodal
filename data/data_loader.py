import torchvision.datasets as dset
import torchvision.transforms as transforms

from transformers import BertTokenizerFast, CLIPImageProcessor
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class CocoDataset(Dataset):
    
    def __init__(self, img_root, ann_path, context_img_root, context_ann_path, transform=None, args=None):
        self.p = args
        self.dataset = dset.CocoCaptions(root=img_root,
                                         annFile=ann_path,
                                         transform=transform
                                        )
        self.context = dset.CocoCaptions(root=context_img_root,
                                         annFile=context_ann_path,
                                         transform=transform
                                        )
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.clip_processor = CLIPImageProcessor()
        # self.tokenizer.add_special_tokens({"additional_special_tokens": ["[IMG]"]})
        self.vocab = self.tokenizer.get_vocab()

        self.in_context_figs, self.in_context_prompts = self.make_in_context_prompt(self.context)
    
    def __getitem__(self, index):
        """Generates in_context images, prompts and labels(captions) for every input.
        """
        image, captions = self.dataset[index]
        
        target = captions[0]
        # caption = "[CLS] " + captions[0] + " [SEP]"
        in_context_images = copy.deepcopy(self.in_context_figs)
        in_context_images.append(image)
        in_context_figs = self.clip_processor.preprocess(in_context_images, return_tensors='pt')['pixel_values'] # shape: [n_shots+1, 3, 224, 224]
        in_context_prompts = copy.deepcopy(self.in_context_prompts)

        inputs = {"images": in_context_figs, "prompts": in_context_prompts}
        return inputs, target
    
    def __len__(self):
        return len(self.dataset)

    def make_prompt(self, cap):
        # make single prompt
        prompt = "Figure:, Caption: " + cap + ";"
        return prompt

    def make_in_context_prompt(self, context):
        # make in context n_shots prompt
        """Generates in context prompts, returns tokenized prompts and figs.
        """
        figs = []
        prompts = []
        for (image, captions) in context:
            figs.append(image)
            prompts.append(self.make_prompt(captions[0]))
        
        # add one more prompt to the end
        prompts.append("Figure:, Caption: ")
        context_prompts = self.in_context_tokenize(prompts)
        return figs, context_prompts
    
    def in_context_tokenize(self, sentences):
        """Tokenize in-context prompts.
        @args sentences: context_prompts, list of prompts
        """
        # cls_id = self.vocab['[CLS]']
        # sep_id = self.vocab['[SEP]']

        beginning_sentence = "[CLS] " + copy.deepcopy(sentences[0])
        sentences[0] = beginning_sentence

        tokenized_sentences = self.tokenizer(sentences, add_special_tokens=False, return_tensors='pt', padding=True, truncation=True)
        # {input_ids: [[101, ...], ..., [...]], type_ids:..., attention_mask:....}
        return tokenized_sentences

    def pad_data(self, caps):
        encoding = self.tokenizer(caps, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])
        # token_type_ids = torch.LongTensor(encoding['token_type_ids'])

        return token_ids, attention_mask, caps
    
    def collate_fn(self, all_data):
        """ all_data: [bs, Example(
                inputs{
                    "images": [n_shots images],
                    "prompts": [n_shots prompts(
                            "token_ids":,
                            "mask":
                        )
                    ]
                },
                targets(captions)
        )]
        """
        image = torch.stack([example[0]["images"] for example in all_data])
        prompts_input_ids = [example[0]["prompts"]['input_ids'] for example in all_data]
        prompts_attention_mask = [example[0]["prompts"]['attention_mask'] for example in all_data]
        max_seq_len = max([p.shape[1] for p in prompts_input_ids])
        # reshape to max_seq_len
        prompts_input_ids = torch.stack([F.pad(p_id, pad=(0,max_seq_len-p_id.shape[1],0,0)) for p_id in prompts_input_ids])
        prompts_attention_mask = torch.stack([F.pad(p_mask, pad=(0,max_seq_len-p_mask.shape[1],0,0)) for p_mask in prompts_attention_mask])
        caption = [example[1] for example in all_data]
        # image, caption = all_data
        token_ids, attention_mask, caption = self.pad_data(caption)
        return {
            'img': image,
            'prompt': {
                "input_ids": prompts_input_ids,
                "attention_mask": prompts_attention_mask
            },
            'cap': caption,
            'token_ids': token_ids,
            'attention_mask': attention_mask,
        }


if __name__ == "__main__":
    # common_trans = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalization 
    # ])
    common_trans = transforms.PILToTensor()
    train_dataset = CocoDataset("resources/in_context", 
                                "resources/annotations/in_context.json", 
                                "resources/in_context", 
                                "resources/annotations/in_context.json", 
                                )

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1, collate_fn=train_dataset.collate_fn)
    for data in list(enumerate(train_dataloader))[:1]:
        print(data)

                                        