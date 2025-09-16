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
