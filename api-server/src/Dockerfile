# pull official base image
FROM python:3.9.16

# set work directory
WORKDIR /src
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# copy project
COPY . /src

# Install Dependancy
RUN pip install -r requirements.txt

CMD ["flask", "run"]
