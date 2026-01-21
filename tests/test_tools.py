import unittest
import pandas as pd
from analyst import StandaloneAnalyst

class TestCheckNormalityTool(unittest.TestCase):
    def setUp(self):
        self.analyst = StandaloneAnalyst()
        self.analyst.context.datasets["test_data"] = pd.DataFrame({
            "normal_col": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })

    def test_check_normality_tool(self):
        result = self.analyst._execute_tool("check_normality", {
            "dataset_name": "test_data",
            "column": "normal_col"
        })
        import json
        self.assertTrue(json.loads(result)['is_normal'])

if __name__ == "__main__":
    unittest.main()
