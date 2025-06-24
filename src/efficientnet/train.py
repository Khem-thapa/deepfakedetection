from data_loader import get_data_generators
from model_builder import build_efficientnet_model


model = build_efficientnet_model(input_shape=(256, 256, 3))


model.compile(
    optimizer='adam', 
    loss='binary_crossentropy', 
    metrics=['accuracy']
    )

train_gen, val_gen = get_data_generators()

model.fit(train_gen, validation_data=val_gen, epochs=5)

# Save the model
model.save('models/efficientnet_model.h5')
