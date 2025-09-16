from typing import List, Tuple

from torch.utils.data import Dataset


class ICDDataset(Dataset):
    """
    datapoints: list of tuples
      (code, justification_list, monitoring_list, evaluation_list,
       assessment_list, treatment_list, label)
    """

    def __init__(self, datapoints: List[Tuple]):
        self.datapoints = datapoints

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        code, just, mon, evl, assm, trt, label = self.datapoints[idx]

        return {
            "code": code,
            "justification": just,
            "monitoring": mon,
            "evaluation": evl,
            "assessment": assm,
            "treatment": trt,
            "label": label
        }


def collate_fn(batch):
    # print(batch)
    # batch is a list of dicts
    codes = [item[0] for item in batch]  # List[str]
    justifications = [item[1] for item in batch]  # List[List[List[str]]]
    monitoring = [item[2] for item in batch]  # List[List[List[str]]]
    evaluation = [item[3] for item in batch]  # List[List[List[str]]]
    assessment = [item[4] for item in batch]  # List[List[List[str]]]
    treatment = [item[5] for item in batch]  # List[List[List[str]]]
    labels = [item[-1] for item in batch]  # List[int]

    return {
        "code": codes,
        "justification": justifications,
        "monitoring": monitoring,
        "evaluation": evaluation,
        "assessment": assessment,
        "treatment": treatment,
        "label": labels,
    }
