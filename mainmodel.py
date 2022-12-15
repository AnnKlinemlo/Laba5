
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

train_dir=r"C:\Users\Аня\Desktop\Work\ISIT\mymodel\train"
test_dir=r"C:\Users\Аня\Desktop\Work\ISIT\mymodel\test"
val_dir=r"C:\Users\Аня\Desktop\Work\ISIT\mymodel\val"

img_width, img_height = 200, 200
input_shape = (img_width, img_height, 3)

batch_size = 20

epochs = 20
# Количество изображений для обучения
nb_train_samples = 2706*4
# Количество изображений для проверки
nb_validation_samples = 460*4
# Количество изображений для тестирования
nb_test_samples = 561*4

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.4)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    subset="training",
    class_mode='categorical')

val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    subset="validation",
    class_mode='categorical')

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


x=train_generator
s_p_e=nb_train_samples // batch_size
v_s=nb_validation_samples // batch_size
model.fit(x,
    steps_per_epoch=s_p_e,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=v_s)

step=nb_test_samples // batch_size

scores = model.evaluate_generator(test_generator, step)

print("Аккуратность на тестовых данных: %.2f%%" % (scores[1]*100))
model.save(r'C:\Users\Аня\Desktop\Work\ISIT\mymodel')
