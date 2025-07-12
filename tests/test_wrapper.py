import unittest

from slither_py import SlitherWrapper

class TestWrapper(unittest.TestCase):
    def test_add(self):
        wrapper = SlitherWrapper()
        self.assertEqual(wrapper.add(2, 3), 5)

if __name__ == '__main__':
    unittest.main()
