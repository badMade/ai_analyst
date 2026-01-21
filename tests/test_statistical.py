import unittest
import pandas as pd
from ai_analyst.tools.statistical import check_normality

class TestStatisticalFunctions(unittest.TestCase):
    def test_check_normality(self):
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = check_normality(series)
        self.assertFalse(result.significant)

if __name__ == "__main__":
    unittest.main()
