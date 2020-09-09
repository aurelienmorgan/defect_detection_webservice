# Uthe Python 3.7.2 container image
FROM python:3.7.2-stretch

# Set the working directory to "app"
WORKDIR /app

# Copy the current directory contents into the container at /app 
ADD . /app

# Install the dependencies
RUN pip install -r requirements.txt


# create an empty "/app/app/media/" folder prior to (volume creation &) parent permission allocation
RUN mkdir -p /app/app/media && \
	chown -R www-data:www-data /app
#RUN stat /app
#RUN cd app && stat media && cd ..

USER www-data

# Run the command to start uWSGI
CMD [ "uwsgi", "app.ini" ]