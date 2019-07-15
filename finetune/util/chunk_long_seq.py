

from finetune.base import BaseModel, PredictMode

def process_long_sequence(X, task, probas=False)
    tasks = ['sequence_labeling', 'classification', 'regression']
    assert task in tasks, 'invalid task for processing long sequences'
    chunk_size = self.config.max_length - 2
    step_size = chunk_size // 3
    if task == 'sequence_labeling':
        arr_encoded = list(itertools.chain.from_iterable(self.input_pipeline._text_to_ids([x]) for x in X))
    else:
        arr_encoded = list(itertools.chain.from_iterable(self.input_pipeline._text_to_ids(x) for x in X))
       
    labels, batch_probas = [], []
    pred_keys = [PredictMode.NORMAL]
    if task != 'regression':
        pred_keys.append(PredictMode.PROBAS)
    for pred in self._inference(X, predict_keys=[PredictMode.PROBAS, PredictMode.NORMAL], n_examples=len(arr_encoded)):
        labels.append(self.input_pipeline.label_encoder.inverse_transform(pred[PredictMode.NORMAL]))
        if task != 'regression':
            batch_probas.append(pred[PredictMode.PROBAS])

    all_subseqs = []
    all_labels = []
    all_probs = []
    all_positions = []

    doc_idx = -1
    for chunk_idx, (label_seq, proba_seq) in enumerate(zip(labels, batch_probas)):
        position_seq = arr_encoded[chunk_idx].char_locs
        start_of_doc = arr_encoded[chunk_idx].token_ids[0][0] == self.input_pipeline.text_encoder.start
        end_of_doc = (
                chunk_idx + 1 >= len(arr_encoded) or
                arr_encoded[chunk_idx + 1].token_ids[0][0] == self.input_pipeline.text_encoder.start
        )
        """
        Chunk idx for prediction.  Dividers at `step_size` increments.
        [  1  |  1  |  2  |  3  |  3  ]
        """
        start, end = 0, None
        if start_of_doc:
            # if this is the first chunk in a document, start accumulating from scratch
            doc_subseqs = []
            doc_labels = []
            doc_probs = []
            doc_positions = []
            doc_starts = []

            doc_idx += 1
            start_of_token = 0
            if not end_of_doc:
                end = step_size * 2
        else:
            if end_of_doc:
                # predict on the rest of sequence
                start = step_size
            else:
                # predict only on middle third
                start, end = step_size, step_size * 2

        if task == 'sequence_labeling':
            label_seq = label_seq[start:end]
            position_seq = position_seq[start:end]
            proba_seq = proba_seq[start:end]

            for label, position, proba in zip(label_seq, position_seq, proba_seq):
                if position == -1:
                    # indicates padding / special tokens
                    continue
                
                # if there are no current subsequence
                # or the current subsequence has the wrong label
                if not doc_subseqs or label != doc_labels[-1] or per_token:
                    # start new subsequence
                    doc_subseqs.append(X[doc_idx][start_of_token:position])
                    doc_labels.append(label)
                    doc_probs.append([proba])
                    doc_positions.append((start_of_token, position))
                    doc_starts.append(start_of_token)
                else:
                    # continue appending to current subsequence
                    doc_subseqs[-1] = X[doc_idx][doc_starts[-1]:position]
                    doc_probs[-1].append(proba)
                start_of_token = position

            if end_of_doc:
                # last chunk in a document
                prob_dicts = []
                for prob_seq in doc_probs:
                    # format probabilities as dictionary
                    probs = np.mean(np.vstack(prob_seq), axis=0)
                    prob_dicts.append(dict(zip(self.input_pipeline.label_encoder.classes_, probs)))
                    if self.multi_label:
                        del prob_dicts[-1][self.config.pad_token]

                all_subseqs.append(doc_subseqs)
                all_labels.append(doc_labels)
                all_probs.append(prob_dicts)
                all_positions.append(doc_positions)

        _, doc_annotations = finetune_to_indico_sequence(raw_texts=X, subseqs=all_subseqs, labels=all_labels,
                                                            probs=all_probs, none_value=self.config.pad_token,
                                                            subtoken_predictions=self.config.subtoken_predictions)

        if per_token:
            return [
                {
                    'tokens': _spacy_token_predictions(
                        raw_text=raw_text,
                        tokens=tokens,
                        probas=probas, 
                        positions=positions
                    ),
                    'prediction': predictions,
                }
                for raw_text, tokens, labels, probas, positions, predictions in zip(
                    X, all_subseqs, all_labels, all_probs, all_positions, doc_annotations
                )
            ]
        else:
            return doc_annotations
    else:
        doc_labels.append(label)
        doc_probs.append(proba)

        if task == 'classification':
                if end_of_doc:
                # last chunk in a document
                mean_pool = np.mean(doc_probs, axis=0)
                pred = np.argmax(mean_pool)
                one_hot = np.zeros_like(mean_pool)
                one_hot[pred] = 1
                label = self.input_pipeline.label_encoder.inverse_transform([one_hot])
                all_labels.append(label)
                all_probs.append(mean_pool)

            if probas:
                return all_probs
            else:
                return all_labels
        else:
                if end_of_doc:
                    mean_pool = np.mean(doc_probs, axis=0)
                    predictions = self.input_pipeline.label_encoder.inverse_transform([mean_pool])
                    all_labels.append(label)
            

