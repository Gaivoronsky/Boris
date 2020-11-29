import numpy as np
import datetime
import smtplib

greet = True


def greeting():
    greeting_lst = ['Привет', 'Приветствую', 'Нихао']
    idx = np.random.randint(1, len(greeting_lst))
    return greeting_lst[idx]


def search_command(command):
    name = {'саш': 'mail', 'коле': 'mail1'}
    global greet
    command = command.lower()
    if greet:
        answer = greeting()
        # sound_text(text)
        greet = False
        return answer

    if 'врем' in command:
        now = datetime.datetime.now()
        answer = f'Сейчас {now.hour} часов {now.minute} минут'
        # sound_text(answer)
        return answer

    if 'сообщ' in command:
        people = [people for people in name.keys() if people in command]
        if people:
            content = command
            # mail = smtplib.SMTP('smtp.gmail.com', 587)
            # mail.ehlo()
            # mail.starttls()
            # mail.login('mail', 'password')
            # mail.sendmail('mail', name[people[0]], content.encode('utf-8'))
            # mail.close()
            return 'Я отправил ваше письмо!'
        else:
            return 'Я не знаю этого человека(('

    if 'сохран' in command:
        return 'Давай'

    return 'Я вас не понял'