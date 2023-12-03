# Машинное обучение ДЗ №1

В рамках домашней работы была разработана модель линейной регрессии, предсказания стоимости автомобиля.
Работа над задачей включала следующие шаги:
- Проведен EDA и обработаны признаки датасета
- Построены графики зависимости признаков и целевой переменной
- Разработана модель только на вещественных признаках
- Были стандартизированы вещественные признаки с помощью `StandardScaler`
- Применена регуляризация
- Разработана модель на вещественных и категориальных признаках
- Закодированы категориальные признаки с помощью подхода `OneHotEncoding`
- Посчитана бизнесовая метрика
- Веса модели, объект `scaler` и порядок колоное для востановления `OneHotEncoding`  были сохранены в `pickle` файлы для дальнейшего использования в рамках сервиса
- Реализован сервис на FastAPI с возможностью отправлять новые объекты в формате json и получать предсказания по ним

Результаты:
- R^2: `0.622069473231696`
- MSE: `203804808290.19186`
- Бизнесовая метрика: `0.248`

Скрины работы сервиса
## /predict_item
![swagger](https://github.com/hotspurs/hse-mlds-ml-hm1/raw/master/images/pred_item_1.jpeg)
![Работа сервиса](https://github.com/hotspurs/hse-mlds-ml-hm1/raw/master/images/pred_item_2.jpeg)
## /predict_items
![swagger](https://github.com/hotspurs/hse-mlds-ml-hm1/raw/master/images/pred_items_1.jpeg)
![Работа сервиса](https://github.com/hotspurs/hse-mlds-ml-hm1/raw/master/images/pred_items_2.jpeg)

В моем случае наибольшей буст в качестве дало добавление категориальных признаков.

Не удалось провести `Feature Engineering`, а так же более точно подобрать оптимальные параметры регуляризации,  веса признаков не занулились.