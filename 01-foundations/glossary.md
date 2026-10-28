# Core AI and ML Glossary

Use this as a quick reference. A term can have more specific meanings in particular research areas; these definitions focus on common introductory usage.

| Term | Plain-language meaning |
|---|---|
| Artificial intelligence | A broad field of systems that perform tasks associated with intelligent behavior |
| Machine learning | Methods that fit patterns from data to make predictions or decisions |
| Deep learning | Machine learning based on neural networks with multiple learned layers |
| Generative model | A model that can generate samples such as text, images, audio, or other data |
| Dataset | A collection of examples used for analysis, training, or evaluation |
| Example / sample | One case in a dataset, such as one customer or one document |
| Feature | An input variable used by a model |
| Target / label | The outcome a supervised model is trained to predict |
| Parameter | A model value learned during training, such as a regression coefficient |
| Hyperparameter | A configuration chosen outside the model's ordinary parameter-learning process |
| Training | The process of adjusting model parameters using data and an objective |
| Inference | Using a trained model to produce outputs for inputs |
| Loss function | A numerical objective used to measure model error during training |
| Metric | A reported measure used to assess model performance |
| Generalization | How well learned behavior works on unseen examples |
| Overfitting | Fitting training-specific details that fail to generalize |
| Underfitting | Failing to capture useful structure in the data |
| Baseline | A simple reference method used to contextualize a more complex method |
| Data leakage | Information improperly crossing from evaluation or future data into model development |
| Embedding | A vector representation intended to encode useful properties of an item |
| Token | A unit used by a tokenizer to represent text for a language model |
| Context window | The amount of token context a model can process in one request |
| Fine-tuning | Further training a model on data for a particular goal or behavior |
| Retrieval-augmented generation | A system pattern that retrieves external material and supplies it to a generative model |
| Hallucination | A generated claim that is unsupported or incorrect, often stated confidently |
| Distribution shift | A change between the data used in development and the data encountered later |

## Quick distinctions

### Parameter vs hyperparameter

A linear regression weight is learned from examples, so it is a parameter. The learning rate used by gradient descent is a hyperparameter. The boundary is method-dependent: some workflows tune values automatically, but the conceptual difference remains useful.

### Loss vs metric

A loss defines the objective the training procedure optimizes. A metric describes performance in a way that is useful to interpret. They may be the same quantity, but they do not have to be. For example, training may use cross-entropy while a report emphasizes recall.

### Training accuracy vs generalization

High training performance tells you how well the model fits observed training examples. It does not, on its own, establish performance on future data. A valid evaluation split and appropriate metrics are essential.

## Self-check

1. Is a feature the same thing as a parameter?
2. Why can a model with lower training loss perform worse on unseen data?
3. Give an example of a hyperparameter.
4. Why is an embedding not necessarily a human-readable explanation?
5. How can retrieval help a language-model application, and what can still go wrong?
