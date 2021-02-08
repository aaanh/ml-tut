from absl import flags, app, logging

import transformers as tr
import nlp

flags.DEFINE_string('model', '', '')

FLAGS = flags.FLAGS

class XLNetDemo():
    def __init__(self):
        super().__init__()
        self.model = tr.XLNetForSequenceClassification.from_pretrained(FLAGS.model)

    def prepare_data(self):
        pass

    def forward():
        pass

    def training_step(self, batch, batch_idx):
        pass


if __name__ == "__main__":


