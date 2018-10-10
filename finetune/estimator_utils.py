import tensorflow as tf


class PatchedParameterServerStrategy(tf.contrib.distribute.ParameterServerStrategy):

    def _verify_destinations_not_different_worker(self, *args, **kwargs):
        # this is currently broken in tf 1.11.0 -- mock this for now
        pass


class LazySummaryHook(tf.train.SummarySaverHook):
    def __init__(self, save_steps=None,
                 save_secs=None,
                 output_dir=None,
                 summary_writer=None):
        super().__init__(save_steps=save_steps, save_secs=save_secs, output_dir=output_dir,
                         summary_writer=summary_writer, scaffold=1)  # scaffold = 1 suppresses exception in __init__

    def _get_summary_op(self):
        """Fetches the summary op either from self._summary_op or self._scaffold.

        Returns:
          Returns a list of summary `Tensor`.
        """
        if self._summary_op is not None:
            summary_op = self._summary_op
        else:
            summary_op = tf.train.Scaffold.get_or_default('summary_op',
                                                           tf.GraphKeys.SUMMARY_OP,
                                                           tf.summary.merge_all)
        if not isinstance(summary_op, list):
            return [summary_op]
        return summary_op
