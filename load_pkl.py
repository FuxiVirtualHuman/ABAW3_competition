def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_word(word_path):
    word_dict = {}
    lists = os.listdir(word_path)
    uni = ['','abramoff', 'acidifying', 'adenoid', 'adenoids', 'adjacency', 'adroop', 'affordances', 'agitators', 'algic',
           'alithea', 'amortize', 'amplitudes', 'anaesthetized', 'archly', 'assortative', 'astronautics', 'audiotape',
           'babbled', 'bactericide', 'bipedalism', 'blaspheming', 'blithely', 'bodach', 'boggley', 'bombsight',
           'borlaug', 'brenta', 'britisher', 'brittler', 'brittles', 'bukovina', 'cantilevered', 'catheterization',
           'champnell', 'charioteer', 'chastened', 'chilmark', 'choppiness', 'chulanont', 'cingulate', 'clattered',
           'claymores', 'cleavers', 'coercing', 'colicky', 'communistic', 'concini', 'conjectures', 'conjunctive',
           'couched', 'creatinine', 'crispness', 'cuttingly', 'deadening', 'deared', 'debroff', 'deciliter',
           'delegitimize', 'deleterious', 'demonstrable', 'deviates', 'devilfish', 'diastolic', 'dickstein',
           'diffident', 'dishonors', 'dispraised', 'doglike', 'dominantly', 'dosia', 'dotcoms', 'easeful', 'ecliptic',
           'ecstatically', 'elegies', 'elfrey', 'emulex', 'enigmatical', 'espaliers', 'espiau', 'eurodollar',
           'exhorting', 'expediently', 'eyecatching', 'eyres', 'fainter', 'fatting', 'fecund', 'fecundity', 'filaments',
           'fineman', 'folkloric', 'fortysomething', 'fragmentary', 'freeboard', 'frizzes', 'fugger', 'fullfledged',
           'gaffany', 'gauzy', 'gebara', 'georgis', 'ghettoes', 'girded', 'girdled', 'globalised', 'gnomon', 'gouraud',
           'grasmick', 'gratulate', 'gravest', 'grayer', 'greying', 'gulfs', 'gyrus', 'haab', 'haltingly', 'handbills',
           'hardener', 'headman', 'heliet', 'hellyer', 'heterosis', 'hominid', 'hued', 'hugin', 'huijin', 'humpbacked',
           'hundredfold', 'hypothesize', 'hypothesizing', 'impassive', 'impulsivity', 'inaugurating', 'interrelated',
           'intubated', 'ivanovitch', 'japanee', 'jasko', 'jidda', 'jinkey', 'kallheim', 'kapolna', 'kilkane', 'kostya',
           'kowtowing', 'kumano', 'kuvera', 'labarre', 'laertes', 'lafite', 'lampson', 'legatum', 'levinas', 'lhamo',
           'lichtman', 'ligands', 'luding', 'lyglenson', 'macora', 'magadha', 'magnesite', 'majesties', 'maltwood',
           'managa', 'massaccio', 'mckelvey', 'medians', 'mediocrities', 'mesenteric', 'mineralization', 'moister',
           'mondamin', 'montagnais', 'mortgagee', 'mousers', 'muchmore', 'mumga', 'mustees', 'myelin', 'nagran',
           'nanchang', 'niihau', 'noncombatant', 'numberplate', 'oberea', 'observables', 'oiliness', 'oilseed', 'okola',
           'orbitofrontal', 'outlives', 'ovate', 'overwatch', 'pagiel', 'panamanians', 'patrimony', 'pendergast',
           'penicillins', 'perceiver', 'perkier', 'pessaries', 'philby', 'phoenicia', 'physiologically', 'piltdown',
           'pinprick', 'plickaman', 'preadolescent', 'precipitancy', 'prerace', 'presuppose', 'proclivities',
           'prologues', 'propitious', 'ptomaines', 'pugilist', 'pulliam', 'quickwitted', 'quillings', 'racicot',
           'rebury', 'refloat', 'retarding', 'rigors', 'ringley', 'roamer', 'roback', 'ruefully', 'ruses', 'sambre',
           'sanlu', 'satiable', 'savoye', 'seamy', 'seepage', 'shagoth', 'shipload', 'shortish', 'shriekers', 'shrikes',
           'sisera', 'slavemaster', 'solyom', 'spasmodic', 'spectres', 'spiderlike', 'spinbronn', 'spinello', 'squibs',
           'stabilised', 'standpoints', 'stereoscope', 'sternness', 'stigmatizing', 'stimulative', 'sunstein',
           'superyachts', 'tabulated', 'tarth', 'thirtytwo', 'tokuda', 'transcriber', 'transcribes', 'trenched',
           'trewin', 'trivialized', 'truisms', 'ugrin', 'ultranationalists', 'umbilicus', 'underestimation',
           'underlies', 'urhobo', 'volkow', 'wafted', 'wakeful', 'wartorn', 'westernized', 'wharfs', 'yokefellow',
           'zammah']
    preload_file = word_path+'.pkl'
    if os.path.exists(preload_file):
        print('load preloaded file:', preload_file)
        return load_pickle(preload_file)

    for l in tqdm(lists):
        f = open(os.path.join(word_path,l),"r")
        lines = f.readlines()
       # word_dict[l] = [k.strip() for k in lines]
        news = []
        for ls in lines:
            ll = ls.strip()
            if "'" in ll:
                t = ll.strip("'")[0].lower()
            else:
                t = ll.lower()
            if t in uni:
                t = "<hashtag>"
            news.append(t)
        word_dict[l] = news
    with open(preload_file, 'wb') as f:
        pickle.dump(word_dict, f)
    return word_dict
def load_mfcc(mfcc_path):
    mfcc_dict = {}
    lists = os.listdir(mfcc_path)
    preload_file = mfcc_path+'.pkl'
    if os.path.exists(preload_file):
        print('load preload file:',preload_file)
        return load_pickle(preload_file)
    for l in tqdm(lists):
        mfcc= np.loadtxt(os.path.join(mfcc_path,l),delimiter=',')
        # mfcc= torch.from_numpy(mfcc).float()
        mfcc_dict[l] = mfcc
    with open(preload_file, 'wb') as f:
        pickle.dump(mfcc_dict, f)
    return mfcc_dict

def load_emb(emb_path):
    emb_dict = {}
    lists = os.listdir(emb_path)
    preload_file = emb_path+'.pkl'
    if os.path.exists(preload_file):
        print('load preload file:',preload_file)
        return load_pickle(preload_file)

    for l in tqdm(lists):
        emb= np.loadtxt(os.path.join(emb_path,l),delimiter=',')
        # mfcc= torch.from_numpy(mfcc).float()
        emb_dict[l] = emb
    with open(preload_file, 'wb') as f:
        pickle.dump(emb_dict, f)
    return emb_dict