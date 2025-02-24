use std::io::Lines;

use burn::{
    config::Config,
    nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, Linear, LinearConfig, Lstm, LstmConfig,
    },
    prelude::Backend,
    tensor::{Int, Tensor},
};

pub struct Seq2Seq<B: Backend> {
    // encoder
    embedding: Embedding<B>,
    dropout: Dropout,
    lstm: Lstm<B>,

    // decoder
    embedding_decoder: Embedding<B>,
    dropout_decoder: Dropout,
    lstm_decoder: Lstm<B>,
    linear_decoder: Linear<B>,
}

impl<B: Backend> Seq2Seq<B> {
    pub fn encoder_forward(&self, input: Tensor<B, 2, Int>) -> burn::nn::LstmState<B, 2> {
        let embedded = self.dropout.forward(self.embedding.forward(input));
        // embedded => length_sequence, N, embedded_size
        let (_, state) = self.lstm.forward(embedded, None);

        state
    }

    pub fn decoder_forward(&self) {
        let input: Tensor<B, 2, Int> = Tensor::from([[0]]);
    }
}

#[derive(Debug, Config)]
pub struct Seq2SeqConfig {
    input: usize,
    hidden: usize,
    output: usize,
    dropout: f64,
}

impl Seq2SeqConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Seq2Seq<B> {
        Seq2Seq {
            // encoder
            embedding: EmbeddingConfig::new(self.input, self.hidden).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            lstm: LstmConfig::new(self.hidden, self.hidden, true).init(device),

            // decoder
            embedding_decoder: EmbeddingConfig::new(self.input, self.hidden).init(device),
            dropout_decoder: DropoutConfig::new(self.dropout).init(),
            lstm_decoder: LstmConfig::new(self.hidden, self.hidden, true).init(device),
            linear_decoder: LinearConfig::new(self.hidden, self.output).init(device),
        }
    }
}
