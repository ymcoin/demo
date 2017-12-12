from dllib import settings

if __name__ == '__main__':
    settings.init('dlkfjadsljfalkdsjflkdsajf;s')
    l = [('a','b'),('c','d'),('e','f')]
    for i,(a,b) in enumerate(l):
        print(i)
        print(a,b)
        print(settings.log_dir)

