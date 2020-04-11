FROM anibali/pytorch:latest

COPY . /app
WORKDIR /app
ENV LANG="es_ES.utf8"
ENV LC_ALL="es_ES.UTF-8"
ENV LC_LANG="es_ES.UTF-8"
ENV PYTHONIOENCODING="utf-8"
RUN sudo chmod -R 777 $PWD

RUN pip install -r requirements.txt

EXPOSE 8080

ENTRYPOINT ["python3"]
CMD ["app.py"]