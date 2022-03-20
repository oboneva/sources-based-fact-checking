import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class NLIModel:
    def __init__(self, hg_model_hub_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            hg_model_hub_name
        )

    @torch.no_grad()
    def get_probs(self, premise: str, hypothesis: str):
        max_length = 256

        tokenized_input_seq_pair = self.tokenizer.encode_plus(
            premise,
            hypothesis,
            max_length=max_length,
            return_token_type_ids=True,
            truncation=True,
        )

        input_ids = (
            torch.Tensor(tokenized_input_seq_pair["input_ids"]).long().unsqueeze(0)
        )
        token_type_ids = (
            torch.Tensor(tokenized_input_seq_pair["token_type_ids"]).long().unsqueeze(0)
        )
        attention_mask = (
            torch.Tensor(tokenized_input_seq_pair["attention_mask"]).long().unsqueeze(0)
        )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=None,
        )
        logits = outputs[0]
        out = torch.softmax(logits, dim=1)
        predicted_probability = out[0].tolist()  # batch_size only one

        return {
            "entailment": predicted_probability[0],
            "neutral": predicted_probability[1],
            "contradiction": predicted_probability[2],
        }


def main():
    model = NLIModel("ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")

    premise = "Two women are embracing while holding to go packages."
    hypothesis = "The men are fighting outside a deli."

    print(model.get_probs(premise, hypothesis))


if __name__ == "__main__":
    main()
