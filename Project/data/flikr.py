from flickrapi import FlickrAPI
import requests
import os
import sys
import time

KEY = 'KEY'
SECRET = 'SECRET'

SIZES = ["url_o", "url_k", "url_h", "url_l", "url_c"]  # in order of preference

def create_folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def get_photos(image_tag):
    extras = ','.join(SIZES)
    flickr = FlickrAPI(KEY, SECRET)
    photos = flickr.walk(text=image_tag,  # it will search by image title and image tags
                            extras=extras,  # get the urls for each size we want
                            privacy_filter=1,  # search only for public photos
                            per_page=50,
                            sort='relevance')  # we want what we are looking for to appear first
    return photos

def get_url(photo):
    for i in range(len(SIZES)):  # makes sure the loop is done in the order we want
        url = photo.get(SIZES[i])
        if url:  # if url is None try with the next size
            return url

def get_urls(image_tag, max):
    photos = get_photos(image_tag)
    counter=0
    urls=[]

    for photo in photos:
        if counter < max:
            url = get_url(photo)  # get preffered size url
            if url:
                urls.append(url)
                counter += 1
            # if no url for the desired sizes then try with the next photo
        else:
            break

    return urls

def download_images(urls, path):
    create_folder(path)  # makes sure path exists

    for url in urls:
        image_name = url.split("/")[-1]
        image_path = os.path.join(path, image_name)

        if not os.path.isfile(image_path):  # ignore if already downloaded
            response=requests.get(url,stream=True)

            with open(image_path,'wb') as outfile:
                outfile.write(response.content)


def main():
    all_species = ['cat', 'lion']
    images_per_species = 200

    for specie in all_species:

        print('Getting urls for', specie)
        urls = get_urls(specie, images_per_species)
        
        print('Downloading images for', specie)
        path = os.path.join('/home/yinghanhuang/Project/data', specie)

        download_images(urls, path)

if __name__ == '__main__':
    main()