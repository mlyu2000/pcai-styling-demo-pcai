# PCAI Styling Demo — frontend (static UI served by nginx)
FROM nginx:1.27-alpine
COPY index.html /usr/share/nginx/html/index.html
# Custom conf: no-store on index.html so browsers never pin a stale UI build.
COPY nginx.conf /etc/nginx/conf.d/default.conf
# nginx:alpine master runs as root to bind :80 and write /var/cache/nginx;
# worker processes drop to the unprivileged 'nginx' user. No USER override here
# (a non-root USER breaks the standard nginx.conf cache/temp dirs).
EXPOSE 80
