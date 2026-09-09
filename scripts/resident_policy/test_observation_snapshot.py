import unittest
from unittest.mock import Mock

from observation_snapshot import read_stable


class SnapshotTests(unittest.TestCase):
    def test_stable_unchanged_object_and_full_request_identity(self):
        observation = object()
        self.assertEqual(read_stable(lambda: observation, lambda: '1:2:3:4', None),
                         ('1:2:3:4', observation))

    def test_previous_or_metadata_only_change_does_not_infer(self):
        load = Mock()
        self.assertIsNone(read_stable(load, lambda: '1:2:3:99', '1:2:3:4'))
        load.assert_not_called()

    def test_changed_read_is_discarded_before_model_can_see_it(self):
        identify = Mock(side_effect=['1:2:3:4', '1:5:6:7', '1:5:6:7', '1:5:6:7'])
        load = Mock(side_effect=['mixed', 'coherent'])
        report = Mock()
        self.assertEqual(read_stable(load, identify, None, report=report, pause=0),
                         ('1:5:6:7', 'coherent'))
        report.assert_called_once()

    def test_continuously_changing_input_fails_closed(self):
        identify = Mock(side_effect=['1:2:3:4', '1:5:6:7'])
        with self.assertRaisesRegex(RuntimeError, 'failed to stabilize'):
            read_stable(lambda: 'mixed', identify, None, timeout=0)

    def test_stable_corruption_is_not_silently_retried(self):
        load = Mock(side_effect=ValueError('bad shape'))
        with self.assertRaisesRegex(ValueError, 'bad shape'):
            read_stable(load, lambda: '1:2:3:4', None)
        load.assert_called_once()

    def test_no_initial_request_is_idle_not_failure(self):
        identify = Mock(side_effect=FileNotFoundError())
        self.assertIsNone(read_stable(Mock(), identify, None, timeout=0))


if __name__ == '__main__':
    unittest.main()
