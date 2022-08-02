# Тестовое задание на позицию Middle MLOps

## Дано:
Разработано несколько ML моделей (для примера даны две - ConstModel и AIModel (src/model.py)) и общий препроцессинг для всех этих моделей (src/preprocessing.py)

В файле sample.py в методе sample1 показано, как это всё работает вместе. Там есть небольшой "хотфикс" на 16-17 строках, который можно будет заменить на адекватное поведение (13 строка) после выполнения первого задания

## Задания
Большинство заданий описаны "не конкретно" - это сделано умышленно, что бы дать простор для выбора конкретного варианта решения проблемы.

Время в скобках ориентировочное, можно не обращать на него внимания

### Задание 1 - Дописать класс BestData [10 минут]
Сейчас BestDataPreparator не работает - в классе BestData не реализованы некоторые методы. 
Необходимо это исправить.

<i>Если выполнить это задание не получится - в последующих задачах можно использовать "хотфикс" на 16-16 строке sample.py</i>

### Задание 2 - Unit-тесты для model.py [20 минут]
Написать 2 или более unit-тестов для классов, приведённых в файле model.py. Код тестов должен находится в папке tests.

Падающие тесты - это хорошо, но следует пометить их таким образом, что бы они не помешали в дальнейшем

<i>Можно использовать pytest или любую другую библиотеку</i>

### Задание 3 - Объединение препроцессинга [30 минут]
Все модели имеют общий препроцессинг и способ запуска. Нужно избавиться от дублирования функционала (вынести общий код в отдельную сущность?)

В идеале, нужно довести всё до такого состояния, что DS разработчику достаточно написаь функцию по типу sample2(data: pd.DataFrame) из sample.py

"Реализовать" инструмент нужно в новом файле src/model-helper.py, показать его применение нужно в файле sample.py на функции sample2

### Задание 4 - CLI [30 минут]
Нужно разработать Command-Line Interface (CLI) для запуска моделей
На вход должно подаваться 2 параметра:
* --id -i Идентификатор модели (соответвие id-модель можно захаркодить)
* --data -d json в текстовом виде (или путь до файла) с данными (формат json показан в sample_data.json) 

Результат этого задания нужно поместить в новый файл main-cli.py, который можно запустить из командной строки (python main-cli.py) 

<i>Можно использовать click или любой другой вариант</i>

### Задание 5 - Базовый CI/CD [20 минут]
Нужно написать 2 bash скрипта:

1) Прогон тестов из папки tests и создание архива build.zip (содержимое: main-cli.py, src, requirements.txt), который будет являться "дистрибутивом". По желанию, можно добавить и другие шаги сборки
2) Разархивирование build.zip в текущую директорию, подготовка окружения и запуск main-cli.py

### Задание 6* - jenkins pipeline [10-30 минут]
(Необязательно)
Разработать jenkinsfile для пайплайна сборки (функционал аналогичен первому скрипту из прошлого задания) + архив должен быть доступен для скачивани c веб-страницы сборки в Jenkins + jenkins отправляет письмо на почту о результатах сборки

### Задание 7 - Базовый API [30-60 минут]
Обернуть модель в мини веб-приложение с простым api (к примеру, через flask):
* /predict/<model_id> - возвращает json с результатом предсказания модели

Где model_id - идентификатор модели из задания 4, а json с данными передаётся в теле запроса

### Задание 8* - Docker [30-60 минут]
(Необязательно)
Собрать docker-контейнер из задания 7

### Задание 9** - Рефакторинг
(Очень необязательно)
При желании, можно порефакторить код "из условия".

