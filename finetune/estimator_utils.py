import tensorflow as tf


class PatchedParameterServerStrategy(tf.contrib.distribute.ParameterServerStrategy):

    def _verify_destinations_not_different_worker(self, *args, **kwargs):
        # this is currently broken in tf 1.11.0 -- mock this for now
        pass


class SaverHookFinetune(tf.train.CheckpointSaverHook):
    def after_create_session(self, session, coord):
        global_step = session.run(self._global_step_tensor)
        self._save(session, global_step)
        self._timer.update_last_triggered_step(global_step)

    def _save(self, session, step):
        tf.logging.info("Saving checkpoints for %d into %s.", step, self._save_path)

        for l in self._listeners:
            l.before_save(session, step)

        self._get_saver().save(session, self._save_path, global_step=step, write_meta_graph=False, write_state=False)

        should_stop = False
        for l in self._listeners:
            if l.after_save(session, step):
                tf.logging.info(
                    "A CheckpointSaverListener requested that training be stopped. "
                    "listener: {}".format(l))
                should_stop = True
        return should_stop
