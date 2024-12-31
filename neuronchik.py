import cv2

# Путь к каскаду (укажите свой путь)
face_cascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')

# Проверка, был ли каскад загружен
if face_cascade.empty():
    print("Ошибка загрузки каскада!")
else:
    print("Каскад загружен успешно!")

# Загрузка изображения
cap = cv2.VideoCapture(0)

while True:
    # Чтение кадра с камеры
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразование в черно-белый формат для улучшения распознавания
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Распознавание лиц на кадре
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Отрисовка прямоугольников вокруг распознанных лиц
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Отображение результата
    cv2.imshow('Face Detection', frame)

    # Выход по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Отключение камеры и закрытие всех окон
cap.release()
cv2.destroyAllWindows()
