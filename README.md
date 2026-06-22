# Legal RAG Hackathon Scaffold

Каркас проекта для хакатона по retrieval/RAG с упором на один основной Colab-ноутбук и хранение всей рабочей логики в `.py`-файлах.

## Структура

```text
.
├── configs/
├── data/
│   ├── raw/
│   └── processed/
├── experiments/
├── notebooks/
│   ├── baseline_reference.ipynb
│   └── main_colab.ipynb
├── outputs/
│   ├── metrics/
│   ├── predictions/
│   └── submissions/
├── scripts/
└── src/
    └── legal_rag/
```

`notebooks/baseline_reference.ipynb` сохранен как исходный референс. Основной workflow должен идти через `notebooks/main_colab.ipynb` и `scripts/*.py`.

## Локальный запуск

1. Создайте окружение и установите зависимости:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Проверьте, что исходные CSV лежат в `data/raw/`:
   `documents.csv`, `train.csv`, `test.csv`, `sample_submission.csv`.

3. Запустите проверку данных:

```bash
python scripts/check_data.py --paths configs/paths.local.yaml
```

4. Запустите baseline:

```bash
python scripts/run_baseline.py --paths configs/paths.local.yaml
```

5. Создайте submission из сохраненных предсказаний:

```bash
python scripts/make_submission.py --paths configs/paths.local.yaml
```

После этого результаты появятся в `outputs/predictions/`, `outputs/metrics/` и `outputs/submissions/`.

## Работа в Colab

Откройте [notebooks/main_colab.ipynb](/Users/a1/Desktop/dev/Legal_RAG_HSE_3/notebooks/main_colab.ipynb) и выполните секции по порядку:

1. Setup
2. Mount Google Drive
3. Clone or pull GitHub repo
4. Install requirements
5. Check data
6. Run baseline
7. Make submission
8. Save outputs to GitHub

Что важно:

- `REPO_URL` в ноутбуке нужно заменить на URL вашего GitHub-репозитория.
- `configs/paths.colab.yaml` ожидает, что данные лежат на Google Drive.
- Код и `outputs/` живут внутри клона репозитория в `/content/legal-rag-hackathon`.
- После рестарта Colab достаточно смонтировать Drive, сделать `git pull` и перейти к нужной секции.

## Где лежат данные и результаты

- Исходные данные: `data/raw/`
- Производные датасеты: `data/processed/`
- Предсказания: `outputs/predictions/`
- Метрики: `outputs/metrics/`
- Сабмиты: `outputs/submissions/`
- Журнал запусков: `experiments/README.md`

`outputs/` специально не добавлен в `.gitignore`, чтобы можно было коммитить важные результаты экспериментов в GitHub.

## Как сохранить outputs в GitHub

Из корня репозитория:

```bash
git status
git add outputs experiments
git commit -m "add experiment outputs"
git push
```

Если коммитить нечего, это нормально: просто продолжайте работу с уже сохраненными файлами.
