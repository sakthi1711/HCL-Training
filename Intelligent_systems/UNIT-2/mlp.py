import tensorflow as tf

# 1. Load Data
# The dataset is loaded and automatically split into training and testing sets.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Preprocess Data
# Normalize pixel values to the range [0, 1] for better convergence.
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. Define the Model (MLP Architecture)
model = tf.keras.Sequential([
    # Input Layer: Flattens the 28x28 image into a 784-element vector.
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    
    # Hidden Layer: A single dense layer with 128 neurons and ReLU activation.
    tf.keras.layers.Dense(128, activation='relu'),
    
    # Output Layer: 10 neurons (for digits 0-9) with Softmax for probability distribution.
    tf.keras.layers.Dense(10, activation='softmax')
])

# 4. Compile the Model
model.compile(optimizer='adam',
              # Loss function for multi-class classification where labels are integers (0, 1, ..., 9).
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# 5. Train the Model
# Train for only 5 epochs to demonstrate quick learning.
print("Training the simple MLP...")
model.fit(x_train, y_train, epochs=5)

# 6. Evaluate
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")