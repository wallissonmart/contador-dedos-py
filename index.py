import cv2
import mediapipe as mp

video = cv2.VideoCapture(0)

hand = mp.solutions.hands
Hand = hand.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

while True:
    check, img = video.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = Hand.process(imgRGB)
    handsPoints = results.multi_hand_landmarks
    h, w, _ = img.shape
    pontos = []

    if handsPoints:
        for points in handsPoints:
            # print(points)
            mpDraw.draw_landmarks(img, points, hand.HAND_CONNECTIONS)
            for id, cord in enumerate(points.landmark):
                cx, cy = int(cord.x * w), int(cord.y * h)
                cv2.putText(img, str(id), (cx, cy + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                pontos.append((cx, cy))

            dedos = [8, 12, 16, 20]
            contador = 0
            if pontos:
                if pontos[4][0] > pontos[3][0]:
                    contador += 1
                for x in dedos:
                    if pontos[x][1] < pontos[x-2][1]:
                        contador += 1

        if (contador == 5):
            print(contador)
            # webbrowser.open(
            # 'https://wol.jw.org/pt/wol/library/r5/lp-t/todas-as-publica%C3%A7%C3%B5es')

    cv2.imshow("Imagem", img)
    cv2.waitKey(1)
