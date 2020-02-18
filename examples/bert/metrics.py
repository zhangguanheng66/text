import collections
import re
import string


def is_qa_exact(a_gold, a_pred):

    def normalize_txt(text):
        # lower case
        text = text.lower()

        # remove punc
        exclude = set(string.punctuation)
        text = "".join(ch for ch in text if ch not in exclude)

        # remove articles
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        text = re.sub(regex, " ", text)

        # white space fix
        return " ".join(text.split())

    return int(normalize_txt(a_gold) == normalize_txt(a_pred))


def compute_f1(ans_pred_tokens_samples):

    '''
        Input: ans_pred_tokens_samples: [(ans1_tokens, pred1_tokens),
                                         (ans2_tokens, pred2_tokens),
                                         ...
                                         (ansn_tokens, predn_tokens)]
        Output: f1 score of the samples
    '''
    def sample_f1(ans_tokens, pred_tokens):
        common = collections.Counter(ans_tokens) & collections.Counter(pred_tokens)
        num_same = sum(common.values())
        if len(ans_tokens) == 0 or len(pred_tokens) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(ans_tokens == pred_tokens)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(ans_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    f1_scores = []
    for (ans_tokens, pred_tokens) in ans_pred_tokens_samples:
        f1_scores.append(sample_f1(ans_tokens, pred_tokens))
