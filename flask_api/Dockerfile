# Uthe Python 3.7.2 container image
FROM python:3.7.2-stretch

# Set the working directory to "api"
WORKDIR /api

# Copy the current directory contents into the container at /api 
ADD . /api

# Install the dependencies
RUN pip install -r requirements.txt



# create an empty "/api/api/media/" folder prior to (volume creation &) parent permission allocation
RUN mkdir -p /api/api/media && \
	chown -R www-data:www-data /api && \
	chown -R www-data:www-data /dev/shm
#RUN stat /api
#RUN cd api && stat media && cd ..

USER www-data

# Command to start Gunicorn
ENTRYPOINT ["./gunicorn_starter.sh"]