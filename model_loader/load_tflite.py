import tensorflow as tf
import keras
import numpy as np


def load_org_model(path):
    model = keras.models.load_model(path)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model, model.predict_on_batch


def load_cps_model(path):
    interpreter = tf.lite.Interpreter(model_path=path)

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    # input_shape = input_details[0]['shape']
    # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

    def predict(inputs):
        outputs = []
        inputs = inputs.astype(np.float32)
        for i in range(0, inputs.shape[0]):
            input_data = inputs[i:i+1, ::]
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
            output_data = interpreter.get_tensor(output_details[0]['index'])
            outputs.append(output_data)
        return np.vstack(outputs)

    return interpreter, predict

