import numpy
from overrides import overrides
from typing import List

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('wikihop_predictor')
class WikiHopPredictor(Predictor):
    """
    predictor interface for WikiHop
    """

    def predict_instance(self, instance: Instance) -> JsonDict:
        """
        Override this method to create a formatted JSON
        :param instance:
        :return:
        """
        output = self._model.forward_on_instance(instance)
        num_cands = len(output['metadata']['choice_text_list'])
        lp = output['label_probs'][:num_cands]
        ans = output['metadata']['choice_text_list'][numpy.argmax(lp)]
        item_id = output['metadata']['id']

        output_json = {
            "id": item_id,
            "answer": ans
        }
        return sanitize(output_json)

    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        output_json_list = []
        for i in range(len(outputs)):
            num_cands = len(outputs[i]['metadata']['choice_text_list'])
            lp = outputs[i]['label_probs'][:num_cands]
            ans = outputs[i]['metadata']['choice_text_list'][numpy.argmax(lp)]
            item_id = outputs[i]['metadata']['id']
            output_json_list.append({
                "id": item_id,
                "answer": ans
            })

        return sanitize(output_json_list)

    @overrides
    def _json_to_instance(self, item_json: JsonDict) -> Instance:
        """
        instatiate the data for the dataset reader
        """
        item_id = item_json["id"]
        docsents = item_json['docsents']
        question = item_json['question']
        candidates = item_json['candidates']
        paths = item_json['paths']
        answer_str = None
        return self._dataset_reader.text_to_instance(item_id, docsents,
                                                     question, candidates, paths,
                                                     answer_str)
