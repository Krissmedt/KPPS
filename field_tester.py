from field import field

settings = {'box':{'xlim':[-1.,1.],'ylim':[-1.,1.],'zlim':[-1.,1.]},
         'resolution':[3]}

f = field(**settings)

print(f.xlim)
print(f.xres)
print(f.nn)