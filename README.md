# Automated Essay Scoring with an Ensemble of Deep-Learning Models

We propose a two-stage model, encompassing two components, namely a prompt-independent and a prompt-dependent component.
The prompt-independent phase implements BERT in a manner contrasting that of the authors’. We use BERT simply as an
encoder, passing the tokenized essay into BERT and passing the output into feed-forward networks. In the
prompt-independent phase, we are able to achieve quadratic weighted kappa (QWK) of **TO-DO**. For the prompt
dependent phase, we experiment with two distinct neural networks: Bi-directional Attention Flow (BiDAF), 
and Hierarchical Attention Networks (HAN), and are able to achieve a QWK of **TO-DO**. We vary the hyperparameters, including
the learning rates, sequence lengths, and tokenization methods, and show the increased robustness of our model. Towards
the end of the paper, we ensemble our models and achieve a quadratic weighted kappa, the official metric, of
**TO-DO**, which is 0.1 points less than the highest score of **TO-DO**.

Our major contributions in the development of this systems are:

- Perused previous literature, handpicking best performing models
- Developed a model with both prompt-independent and prompt-dependent components
- Compared our model's performance against previously developed models 
- Incorporated modern encoders, like BERT, and modern concepts, like attention, into our models
- Performed ablation studies to prove the efficacy of our models