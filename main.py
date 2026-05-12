import os
import subprocess
import sys
from dotenv import load_dotenv, dotenv_values, set_key


def get_env_path():
    base_path = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(base_path, ".env")
    return env_path, base_path

env_path, base_path = get_env_path()
load_dotenv(env_path)

def save_and_reload():
    global env_path
    for key in list(os.environ.keys()):
        if key.startswith('X-API-KEY'):
            os.environ.pop(key, None)
    load_dotenv(env_path, override=True)

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def mask_key(key):
    if not key or len(key) < 8:
        return "***"
    return key[:4] + "*" * (len(key) - 8) + key[-4:]

def get_api_keys():
    keys = []
    i = 0
    while True:
        key = os.getenv(f'X-API-KEY{i}')
        if key is None:
            break
        keys.append(key)
        i += 1
    return keys

def manage_keys():
    while True:
        clear_screen()
        print("=== Управление API ключами ===\n")
        keys = get_api_keys()
        if keys:
            print("Добавленные ключи:")
            for i, k in enumerate(keys):
                print(f"{i + 1}. {mask_key(k)}")
        else:
            print("Нет добавленных ключей.")
        print("\n1. Добавить ключ")
        print("2. Удалить ключ")
        print("3. Назад")
        choice = input("\nВыберите действие: ").strip()
        if choice == "1":
            values = dotenv_values(env_path)
            new_key = input("Введите новый API-ключ: ").strip()
            if new_key not in values.values():
                set_key(env_path, f'X-API-KEY{len(keys)}', new_key)
                save_and_reload()
                print("Ключ добавлен!")
                save_and_reload()
            else:
                print('Такой ключ уже есть')
        elif choice == "2":
            if not keys:
                input("Нет ключей для удаления...")
                continue
            try:
                num = int(input("Введите номер ключа для удаления: ")) - 1
                if 0 <= num < len(keys):
                    values = dotenv_values(env_path)
                    with open(env_path, 'w', encoding='utf-8') as f:
                        new_idx = 0
                        for key, value in values.items():
                            if key.startswith('X-API-KEY'):
                                try:
                                    k_idx = int(key.replace('X-API-KEY', ''))
                                    if k_idx != num:
                                        f.write(f'X-API-KEY{new_idx}={value}\n')
                                        new_idx += 1
                                except:
                                    f.write(f'{key}={value}\n')
                            else:
                                f.write(f'{key}={value}\n')
                    print("Ключ удалён.")
                    save_and_reload()
                else:
                    print("Неверный номер!")
            except:
                print("Ошибка ввода!")
        elif choice == "3":
            break
        else:
            print("Неверный выбор!")
        input("\nНажмите Enter для продолжения...")

def run_script(script_name: str, folder: str):
    script_path = os.path.join(base_path, folder, script_name)

    print(f"Попытка запуска: {script_path}")

    if not os.path.exists(script_path):
        print(f"Ошибка: Файл не найден!\n{script_path}")
        print(f"Текущая папка: {base_path}")
        input("\nНажмите Enter...")
        return

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.join(base_path, folder),  # Важно!
            check=False,
            capture_output=False,
            text=True
        )
        if result.returncode != 0:
            print(f"Скрипт завершился с кодом {result.returncode}")
    except Exception as e:
        print(f" Критическая ошибка запуска: {e}")

    input("\nНажмите Enter для возврата в меню...")

def main_menu():
    while True:
        clear_screen()
        print("===== Kinopoisk Analyzer =====\n")
        print("1. Управление API ключами")
        print("2. Сбор данных")
        print("3. Анализ данных")
        print("4. Выход")
        choice = input("\nВыберите действие: ").strip()
        if choice == "1":
            manage_keys()
        elif choice == "2":
            print("Запуск сбора данных...\n")
            run_script('api.py', 'api')
        elif choice == "3":
            print("Запуск анализа...\n")
            run_script('analyze.py', 'analyze')
        elif choice == "4":
            print("До свидания!")
            break
        else:
            print("Неверный выбор!")

if __name__ == "__main__":
    main_menu()
