FROM python:3.9.13-slim-buster

# set the working directory and copy files

WORKDIR /api

COPY requirements.txt .
COPY ./src /api/src
# set env variables

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# upgrade pip & install dependencies
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

# set workdir to src
WORKDIR /api/src

# run the application
EXPOSE 8000
CMD ["python", "-m", "main"]
