# Stage 1: Build
FROM node:16-alpine as build
WORKDIR /app

COPY package.json yarn.lock ./
RUN yarn install

COPY . ./
RUN yarn build

# Stage 2: Serve
FROM node:16-alpine
WORKDIR /app

COPY --from=build /app ./
EXPOSE 3000

CMD ["yarn", "start"]
