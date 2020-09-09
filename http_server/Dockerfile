# Use the Nginx image
FROM nginx

# Remove the default nginx.conf
RUN rm /etc/nginx/conf.d/default.conf

#Replace with our own nginx.conf
COPY nginx.conf /etc/nginx/conf.d/


#makes the www-data user the owner and group for a few paths that Nginx will need to write to.
#The first is the PID file /var/run/nginx.pid. Since this file doesn’t already exist,
#use the 'touch' command to create an empty file before setting the ownership.
#The second path is a directory Nginx uses for various caches.
#The second path is a directory Nginx uses for temporary files.
RUN touch /var/run/nginx.pid && \
  chown -R www-data:www-data /var/run/nginx.pid && \
  chown -R www-data:www-data /var/cache/nginx && \
  chown -R www-data:www-data /etc/nginx && \
  chown -R www-data:www-data /var/log/nginx/

# remove the user nginx; line to avoid a warning
# since this directive is only meaningfull when Nginx is running as root.
RUN sed -i 's/user  nginx;/ /g' /etc/nginx/nginx.conf


##############################################
# DEV PLATFORM (running Docker on Oracle VM) #
# https://stackoverflow.com/questions/6236078/how-to-clear-the-cache-of-nginx
# https://serverfault.com/questions/269420/disable-caching-when-serving-static-files-with-nginx-for-development
##############################################
# sets "sendfile" to "off" in the main conf file
RUN sed -i 's/sendfile        on/sendfile off/g' /etc/nginx/nginx.conf
RUN cat /etc/nginx/nginx.conf
# clear the cache
RUN rm -rf /var/cache/nginx/*
##############################################




# create empty "/home/flask_app/static/" & "/home/flask_app/static/" folders
# prior to volume creation (thus allowing for a proper user permission assignment)
RUN touch /var/run/nginx.pid && \
  mkdir -p /home/web_app/static && \
  chown -R www-data:www-data /home/web_app/static/ && \
  mkdir -p /home/web_app/media && \
  chown -R www-data:www-data /home/web_app/media


USER www-data