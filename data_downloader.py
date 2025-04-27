from bing_image_downloader import downloader

# Add names here to download new Bollywood celebrities
actors = ['Shah Rukh Khan', 'Deepika Padukone', 'Ranveer Singh', 'Kiara Advani', 'Vicky Kaushal']

for actor in actors:
    downloader.download(actor, limit=100, output_dir='dataset', adult_filter_off=True, force_replace=False, timeout=60, verbose=True)
