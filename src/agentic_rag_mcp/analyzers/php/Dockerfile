FROM php:8.2-cli

RUN apt-get update && apt-get install -y --no-install-recommends \
        unzip curl git \
    && rm -rf /var/lib/apt/lists/*

# Install Composer
RUN curl -sS https://getcomposer.org/installer \
    | php -- --install-dir=/usr/local/bin --filename=composer

WORKDIR /analyzer

COPY composer.json .
RUN composer install --no-dev --no-interaction --prefer-dist

COPY analyze.php .

ENTRYPOINT ["php", "analyze.php"]
