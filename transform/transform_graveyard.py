# Old functions for calculating attributes
import json
from datetime import datetime
from typing import Tuple
import numpy as np

# {"user_id":103,"name":"Paweł Kopyto","city":"Wągrowiec","street":"Zielna 933","favourite_genres":["motown","mpb","brill building pop"],"premium_user":false}
# {"id":"3sRHiG3zQZwJkWBI6GjMQF","artist_id":"0PN0fbe41KbuzlRYnoajNm","name":"Canción De Un Preso","popularity":32,"duration_ms":226239,"explicit":0,"release_date":"1991-12-19","danceability":0.574,"energy":0.132,"key":5,"mode":1,"loudness":-11.335,"speechiness":0.0455,"acousticness":0.821,"instrumentalness":1.46e-6,"liveness":0.255,"valence":0.633,"tempo":110.734,"time_signature":3}
# {"id":"2Sqr0DXoaYABbjBo9HaMkM","name":"Sara Bareilles","genres":["acoustic pop","dance pop","lilith","neo mellow","pop","pop rock","post-teen pop"]}
# {"timestamp":"2020-11-19T07:06:15.915000+02","user_id":1574,"track_id":"29b70iOlgRkYQ8ZzINZXph","event_type":"Play","session_id":39084}

PLAY_WEIGHT = 2
LIKE_WEIGHT = 10
SKIP_WEIGHT = -1
EARNINGS_FILE_COUNT = 4
UNITS_FILE_COUNT = 42
SPECIAL_CITIES = {
    "Jastrzębie-Zdrój": 8987.51,
    "Warszawa": 9625.74,
    "Wałbrzych": 7584.09,
    "Stargard Szczeciński": 6404.24
} # cities that fail automatic earnings search
TRACK_PARAMETERS = ["popularity", "duration_ms", "explicit", "release_date", "danceability", "energy", "key", "mode",
    "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "time_signature"]
SCALAR_ATTRIBUTES = ['premium', 'sex', 'salary', 'account_age', 'sessions', 'sessions_daily', 'time_spent_s',
    'time_spent_daily_s', 'time_spent_per_session_s', 'ads_displayed', 'ads_watched', 'ads_watch_rate',
    'total_weight', 'tracks_listened', 'plays', 'likes', 'skips']
TRACK_NONZERO_GENRES = ['none_of_the_specified', '"black \'n\' roll"', '"death \'n\' roll"', 'accordeon', 'accordion', 'acid rock', 'acoustic blues', 'acoustic punk', 'acoustic rock', 'adult standards', 'african-american classical', 'afrofuturism', 'alabama indie', 'album rock', 'alt z', 'alternative country', 'alternative dance', 'alternative metal', 'alternative pop', 'alternative pop rock', 'alternative rock', 'alternative roots rock', 'ambient black metal', 'american folk revival', 'american modern classical', 'anti-folk', 'appalachian folk', 'arab folk', 'arab pop', 'art pop', 'art punk', 'art rock', 'athens indie', 'atmospheric black metal', 'austindie', 'australian alternative rock', 'australian psych', 'australian rock', 'austrian classical piano', 'autonomous black metal', 'avant-garde jazz', 'baiao', 'bal-musette', 'ballroom', 'bandolim', 'bandoneon', 'bangla', 'baroque', 'baroque pop', 'barrelhouse piano', 'bath indie', 'bay area indie', 'beatlesque', 'bebop', 'belly dance', 'bhajan', 'bhojpuri pop', 'big band', 'big beat', 'big room', 'birmingham metal', 'black death', 'black metal', 'black thrash', 'blackened crust', 'bluegrass', 'bluegrass gospel', 'blues', 'blues rock', 'bomba y plena', 'boogie-woogie', 'bossa nova', 'boston rock', 'bouzouki', 'bow pop', 'brass ensemble', 'brazilian black metal', 'brazilian groove metal', 'brazilian metal', 'brazilian power metal', 'brazilian progressive metal', 'brazilian thrash metal', 'brighton indie', 'brill building pop', 'british black metal', 'british blues', 'british dance band', 'british death metal', 'british folk', 'british grindcore', 'british indie rock', 'british invasion', 'british jazz', 'british power metal', 'british soundtrack', 'britpop', 'brooklyn indie', 'brutal death metal', 'bubble trance', 'bubblegum pop', 'buffalo ny metal', 'bulgarian electronic', 'bulgarian pop', 'bulgarian rock', 'c-pop', 'c86', 'cabaret', 'cajun', 'california hardcore', 'calypso', 'canadian blues', 'canadian country', 'canadian indie', 'canadian metal', 'canadian pop', 'canadian rock', 'canadian singer-songwriter', 'candy pop', 'canterbury scene', 'canzone genovese', 'canzone napoletana', 'carnatic', 'carnatic vocal', 'carnaval cadiz', 'celtic rock', 'chamame', 'chamber pop', 'chamber psych', 'chanson', 'chicago blues', 'chicago indie', 'chicago rap', 'chilean rock', 'choro', 'christian rock', 'chutney', 'circuit', 'classic arab pop', 'classic bollywood', 'classic canadian rock', 'classic colombian pop', 'classic eurovision', 'classic finnish pop', 'classic finnish rock', 'classic garage rock', 'classic greek pop', 'classic hardstyle', 'classic icelandic pop', 'classic iskelma', 'classic italian folk pop', 'classic italian pop', 'classic j-pop', 'classic kollywood', 'classic pakistani pop', 'classic rock', 'classic schlager', 'classic soul', 'classic soundtrack', 'classic tollywood', 'classic uk pop', 'classical', 'classical organ', 'classical performance', 'classical piano', 'classical tenor', 'classical trumpet', 'comedy rock', 'comic', 'comic metal', 'concepcion indie', 'conscious hip hop', 'contemporary classical', 'contemporary post-bop', 'cool jazz', 'copla', 'cosmic american', 'country', 'country blues', 'country gospel', 'country rock', 'cowboy western', 'cowpunk', 'croatian electronic', 'crossover thrash', 'cumbia', 'cumbia chilena', 'cumbia ranchera', 'cyberpunk', 'dance pop', 'dance rock', 'dance-punk', 'danish electronic', 'danish metal', 'danish rock', 'dansktop', 'dark black metal', 'dark minimal techno', 'dayton indie', 'death metal', 'deathgrind', 'deathrash', 'deep adult standards', 'deep deep house', 'deep deep tech house', 'deep disco house', 'deep dnb', 'deep groove house', 'deep liquid bass', 'deep new americana', 'deep uplifting trance', 'delta blues', 'desi pop', 'destroy techno', 'detroit rock', 'dinner jazz', 'disco', 'disney', 'diva house', 'dixieland', 'doo-wop', 'doom metal', 'double drumming', 'dream pop', 'dreamo', 'drift', 'dungeon synth', 'dusseldorf electronic', 'dutch house', 'dutch tech house', 'dutch trance', 'early modern classical', 'early music', 'early us punk', 'easy listening', 'edm', 'electric blues', 'electro house', 'electro jazz', 'electronic rock', 'electronica', 'electropop', 'electropowerpop', 'emo', 'english indie rock', 'enka', 'entehno', 'epic doom', 'euphoric hardstyle', 'eurobeat', 'eurodance', 'eurovision', 'exotica', 'experimental', 'fado', 'filmi', 'finnish black metal', 'finnish blues', 'finnish death metal', 'finnish doom metal', 'finnish hard rock', 'finnish heavy metal', 'finnish melodeath', 'finnish metal', 'finnish power metal', 'finnish progressive metal', 'finnish tango', 'florida death metal', 'flute rock', 'folclore salteno', 'folk', 'folk black metal', 'folk metal', 'folk rock', 'folklore argentino', 'forro', 'forro tradicional', 'frankfurt electronic', 'freak folk', 'freakbeat', 'free improvisation', 'free jazz', 'french death metal', 'french hip hop', 'french jazz', 'french metal', 'french movie tunes', 'french pop', 'french post-punk', 'french soundtrack', 'frenchcore', 'full on', 'funeral doom', 'funk', 'funk metal', 'funk rock', 'funky tech house', 'gabba', 'garage rock', 'garage rock revival', 'gauze pop', 'german baroque', 'german electronica', 'german hard rock', 'german heavy metal', 'german jazz', 'german metal', 'german oi', 'german power metal', 'german punk', 'german rock', 'german thrash metal', 'ghazal', 'ghettotech', 'girl group', 'glam metal', 'glam punk', 'glam rock', 'goregrind', 'gothenburg metal', 'gothic black metal', 'gothic metal', 'gothic symphonic metal', 'greek black metal', 'greek clarinet', 'greek indie', 'greek metal', 'greek pop', 'grindcore', 'grisly death metal', 'groove metal', 'grunge', 'guitarra argentina', 'gypsy jazz', 'haitian traditional', 'happy hardcore', 'hard bop', 'hard rock', 'hardcore punk', 'hardcore techno', 'hardstyle', 'harlem renaissance', 'harmonica blues', 'heartland rock', 'hengelliset laulut', 'hindustani classical', 'hindustani vocal', 'hip hop', 'house', 'houston rap', 'icelandic pop', 'icelandic traditional', 'indian classical', 'indie anthem-folk', 'indie electronica', 'indie folk', 'indie fuzzpop', 'indie pop', 'indie poptimism', 'indie rock', 'indietronica', 'industrial', 'industrial metal', 'industrial rock', 'instrumental funk', 'instrumental rock', 'irish rock', 'italian adult pop', 'italian black metal', 'italian death metal', 'italian metal', 'italian progressive rock', 'italian tech house', 'italian tenor', 'italian trance', 'j-metal', 'j-rock', 'jam band', 'jamgrass', 'jangle pop', 'japanese heavy metal', 'jazz', 'jazz accordion', 'jazz blues', 'jazz clarinet', 'jazz drums', 'jazz funk', 'jazz fusion', 'jazz guitar', 'jazz house', 'jazz piano', 'jazz rock', 'jazz saxophone', 'jazz trombone', 'jazz trumpet', 'jazz vibraphone', 'jazz violin', 'jewish cantorial', 'judaica', 'jug band', 'jump blues', 'kashmiri pop', 'kingston on indie', 'kleine hoerspiel', 'klezmer', 'la indie', 'laiko', 'latin', 'latin classical', 'latin hip hop', 'latin jazz', 'latin metal', 'latin soundtrack', 'lebanese pop', 'leicester indie', 'light music', 'lilith', 'liverpool indie', 'lo star', 'lo-fi', 'louisiana blues', 'louisiana metal', 'louisville indie', 'lounge', 'macedonian folk', 'madchester', 'mambo', 'manchester indie', 'marathi traditional', 'melancholia', 'mellow gold', 'melodic black metal', 'melodic death metal', 'melodic groove metal', 'melodic hard rock', 'melodic metal', 'melodic metalcore', 'melodic rap', 'melodic thrash', 'memphis blues', 'memphis soul', 'merseybeat', 'metal', 'metal catarinense', 'metal guitar', 'metal mineiro', 'metalcore', 'mexican classic rock', 'mexican metal', 'middle eastern traditional', 'minimal dub', 'minimal tech house', 'mod revival', 'modern alternative rock', 'modern blues', 'modern blues rock', 'modern bollywood', 'modern folk rock', 'modern hard rock', 'modern power pop', 'modern rock', 'modern southern rock', 'motown', 'movie tunes', 'musica tradicional cubana', 'musica tradicional dominicana', 'musiikkia lapsille', 'muzica populara', 'nagpuri pop', 'nashville sound', 'native american', 'neo classical metal', 'neo mellow', 'neo-progressive', 'neo-psychedelic', 'neo-rockabilly', 'neo-trad metal', 'neoclassicism', 'neon pop punk', 'neue deutsche harte', 'new age piano', 'new americana', 'new jersey indie', 'new orleans blues', 'new orleans jazz', 'new rave', 'new romantic', 'new wave', 'new wave of thrash metal', 'new wave pop', 'new york death metal', 'newcastle indie', 'no wave', 'noise pop', 'noise rock', 'nordic folk metal', 'north east england indie', 'northern irish indie', 'northern soul', 'norwegian black metal', 'norwegian classical', 'norwegian death metal', 'norwegian metal', 'nottingham indie', 'nouvelle chanson francaise', 'nu age', 'nu gaze', 'nu metal', 'nueva cancion', 'nuevo tango', 'nwobhm', 'nwothm', 'nz dnb', 'nz hip hop', 'okc indie', 'oklahoma country', 'old school rap francais', 'old school thrash', 'old-time', 'ontario indie', 'opera metal', 'oratory', 'orquesta tipica', 'otacore', 'oud', 'oulu metal', 'outlaw country', 'oxford indie', 'pagan black metal', 'paisley underground', 'pakistani pop', 'palm desert scene', 'parody', 'permanent wave', 'philly rap', 'philly soul', 'piano blues', 'piano rock', 'piedmont blues', 'pirate', 'pittsburgh rap', 'pixie', 'poetry', 'polish black metal', 'polish death metal', 'polish metal', 'polish prog', 'political hip hop', 'pop', 'pop chileno', 'pop dance', 'pop edm', 'pop emo', 'pop punk', 'pop rap', 'pop rock', 'porro', 'portuguese black metal', 'portuguese metal', 'portuguese rock', 'post-grunge', 'post-hardcore', 'post-punk', 'post-teen pop', 'power metal', 'power pop', 'power thrash', 'pre-war blues', 'progressive black metal', 'progressive bluegrass', 'progressive breaks', 'progressive death metal', 'progressive doom', 'progressive electro house', 'progressive groove metal', 'progressive house', 'progressive metal', 'progressive power metal', 'progressive rock', 'progressive trance', 'progressive trance house', 'progressive uplifting trance', 'protopunk', 'psybreaks', 'psychedelic blues-rock', 'psychedelic folk', 'psychedelic rock', 'psychedelic soul', 'psychedelic trance', 'pub rock', 'punjabi folk', 'punk', 'punk blues', 'quebec death metal', 'quebec indie', 'rabindra sangeet', 'ragtime', 'rap', 'rap inde', 'rap metal', 'rap rock', 'rawstyle', 're:techno', 'rebetiko', 'reggae fusion', 'reggaeton', 'reggaeton colombiano', 'rhythm and blues', 'riot grrrl', 'rochester ny indie', 'rock', 'rock drums', 'rock-and-roll', 'rockabilly', 'romanian rock', 'roots rock', 'russian folk', 'russian modern classical', 'russian trance', 'ryukoka', 'sacramento indie', 'samba', 'samba-jazz', 'sambass', 'scorecore', 'scottish indie', 'scottish metal', 'scottish new wave', 'scottish rock', 'scottish singer-songwriter', 'screamo', 'seattle metal', 'sertanejo tradicional', 'shanty', 'sheffield indie', 'shimmer pop', 'shoegaze', 'shred', 'singer-songwriter', 'ska', 'skate punk', 'slayer', 'sleaze rock', 'sleep', 'sludge metal', 'socal pop punk', 'soft rock', 'son cubano', 'son cubano clasico', 'sophisti-pop', 'soul', 'soul blues', 'soul jazz', 'soundtrack', 'south african hip hop', 'south african pop', 'south african pop dance', 'southern hip hop', 'southern metal', 'southern rock', 'space age pop', 'space rock', 'speed metal', 'speedcore', 'spiritual jazz', 'steel guitar', 'stomp and holler', 'stomp pop', 'stoner metal', 'stoner rock', 'straight-ahead jazz', 'stride', 'string band', 'sufi', 'sunset lounge', 'sunshine pop', 'supergroup', 'swamp blues', 'swamp pop', 'swamp rock', 'swedish alternative rock', 'swedish black metal', 'swedish country', 'swedish death metal', 'swedish doom metal', 'swedish garage rock', 'swedish hard rock', 'swedish heavy metal', 'swedish indie rock', 'swedish jazz', 'swedish melodic rock', 'swedish metal', 'swedish power metal', 'swedish progressive metal', 'swing', 'swing italiano', 'swiss black metal', 'swiss metal', 'swiss rock', 'symphonic black metal', 'symphonic death metal', 'symphonic metal', 'symphonic rock', 'synthpop', 'syrian pop', 'tamil pop', 'tango', 'tango cancion', 'tape club', 'tech house', 'tech trance', 'technical black metal', 'technical death metal', 'technical groove metal', 'technical melodic death metal', 'technical thrash', 'techno', 'tempe indie', 'texas blues', 'texas country', 'thrash core', 'thrash metal', 'tin pan alley', 'tolkien metal', 'torch song', 'traditional blues', 'traditional country', 'traditional folk', 'trance', 'trap', 'trap latino', 'trap queen', 'tribal house', 'tropical', 'tropical house', 'turkish classical', 'turkish folk', 'turkish jazz', 'turkish pop', 'twee indie pop', 'uk americana', 'uk dnb', 'uk doom metal', 'uk metalcore', 'uk pop', 'uk post-punk', 'uplifting trance', 'uptempo hardcore', 'us power metal', 'vancouver metal', 'vaudeville', 'vegas indie', 'velha guarda', 'vietnamese pop', 'viking black metal', 'viking metal', 'vintage chanson', 'vintage chinese pop', 'vintage finnish jazz', 'vintage gospel', 'vintage hollywood', 'vintage italian pop', 'vintage jazz', 'vintage old-time', 'vintage schlager', 'vintage spanish pop', 'vintage swedish pop', 'vintage swing', 'vintage taiwan pop', 'vintage tango', 'violao', 'violin', 'virginia metal', 'virginia punk', 'vocal harmony group', 'vocal jazz', 'volksmusik', 'washboard', 'washington indie', 'welsh rock', 'western swing', 'world chill', 'wrestling', 'xtra raw', 'yacht rock', 'ye ye', 'yodeling', 'zolo', 'zydeco']
# with artificial none_of_the_specified

def find_powiat_id(city: str) -> str:
    for i in range(UNITS_FILE_COUNT):
        with open(f"../data/units/units{i}.json", "r") as units_file:
            units = json.load(units_file)
            for unit in units["results"]:
                if unit["name"] == city:
                    return unit["parentId"]

def find_earnings(powiat_id: str) -> float:
    for i in range(EARNINGS_FILE_COUNT):
        with open(f"../data/earnings/earnings{i}.json", "r") as earnings_file:
            earnings = json.load(earnings_file)
            for earning in earnings["results"]:
                if earning["id"] == powiat_id:
                    return earning["values"][0]["val"]

def find_salary(city: str) -> float:
    if city in SPECIAL_CITIES.keys():
        return SPECIAL_CITIES[city]
    earnings = find_earnings(find_powiat_id(city))
    if earnings is None or earnings == 0:
        print(f"Earnings in {city} are {earnings}")
        return 0.0

    return earnings

def to_sex(name: str) -> int:
    first_name = name.split()[0]
    if first_name[-1] == "a":
        return 0
    return 1

def to_datetime(timestamp: str) -> datetime:
    return datetime.strptime(timestamp[0:-3], "%Y-%m-%dT%H:%M:%S.%f")
    # timezone cut off - Python can't process this format easily and it's not necessary

def load_tracks() -> dict:
    dictionary = dict()
    with open("../data/tracks.jsonl", "r") as track_files:
        for line in track_files:
            track = json.loads(line)
            dictionary[track["id"]] = track
    return dictionary

def load_artists_genres() -> Tuple[dict, list]:
    artists = dict()
    genres = set()
    with open("../data/artists.jsonl", "r") as artist_files:
        for line in artist_files:
            artist = json.loads(line)
            artists[artist["id"]] = artist

            for genre in artist["genres"]:
                genres.add(genre)

    genres = list(genres)
    genres.sort()
    return artists, genres

def get_user_genres() -> list:
    genres = set()
    with open("../data/users.jsonl", "r") as users_file:
        for line in users_file:
            genres.update(json.loads(line)["favourite_genres"])
    genres = list(genres)
    genres.sort()
    return genres

def process_track_interaction(interaction: dict, user: dict, tracks: dict, artists: dict, genres: list) -> None:
    if interaction["event_type"] == "Play":
        user["plays"] += 1
        weight = PLAY_WEIGHT

        # FIXME: [optional] measure time preferences with play time (considering skips/end of session) instead of count
        hour = to_datetime(interaction["timestamp"]).hour
        user["time_preferences"][hour] += 1
    elif interaction["event_type"] == "Like":
        user["likes"] += 1
        weight = LIKE_WEIGHT
    else:
        user["skips"] += 1
        weight = SKIP_WEIGHT
    user["total_weight"] += weight

    track = tracks[interaction["track_id"]]
    artist = artists[track["artist_id"]]

    for i, parameter in enumerate(TRACK_PARAMETERS):
        if parameter == "release_date":
            user["track_preferences"][i] += int(track["release_date"][0:4]) * weight
        else:
            user["track_preferences"][i] += (track[parameter] or 0.5) * weight # FIXME: null mode?

    genre_count = len(artist["genres"])
    for i, genre in enumerate(genres):
        if genre in artist["genres"]:
            user["genre_preferences"][i] += weight / genre_count

    # TODO: maybe normalize everything whenever sum of weights exceeds 100 (for float precision)
    # TODO: check min/max parameter values in tracks to normalize

def preprocess_users(users: dict, len_genres: int):
    with open("../data/users.jsonl", "r") as users_file:
        for line in users_file:
            user_json = json.loads(line)
            user = dict()
            user["sex"] = to_sex(user_json["name"])
            user["salary"] = find_salary(user_json["city"])
            user["premium"] = 1 if user_json["premium_user"] is True else 0

            user["account_age"] = -1
            user["sessions"], user["time_spent_s"], user["ads_displayed"], user["ads_watched"], user["total_weight"], \
                user["tracks_listened"], user["plays"], user["likes"], user["skips"] = 0, 0, 0, 0, 0, 0, 0, 0, 0
            user["genre_preferences"] = [0 for _ in range(len_genres)]
            user["track_preferences"] = [0 for _ in range(len(TRACK_PARAMETERS))]
            user["time_preferences"] = [0 for _ in range(24)]

            users[user_json["user_id"]] = user

def process_sessions(users: dict, tracks: dict, artists: dict, genres: list) -> int:
    with open("../data/sessions.jsonl", "r") as sessions_file:
        first_line = True
        for line in sessions_file:
            if first_line:
                # fake first interaction to be moved to old
                first_line = False
                interaction = json.loads(line)
                interaction["event_type"] = "Play"
                interaction["user_id"] = 2819
                first_date = to_datetime(interaction["timestamp"])
                session_start = first_date

            old = interaction
            old_user = users[old["user_id"]]
            interaction = json.loads(line)
            user = users[interaction["user_id"]]
            date = to_datetime(interaction["timestamp"])

            if old["session_id"] != interaction["session_id"]: # new session detected
                old_user["time_spent_s"] += (to_datetime(old["timestamp"]) - session_start).seconds
                old_user["sessions"] += 1
                # FIXME: [optional] this method is simplified and cuts off the time of the last interaction

                session_start = date
            else: # no new session - didn't quit on ad
                if old["event_type"] == "Advertisement":
                    old_user["ads_watched"] += 1

            if user["account_age"] == -1:
                user["account_age"] = (date - first_date).days
                # actually saves days since first interaction in file, it gets transformed later

            if interaction["event_type"] == "Advertisement":
                user["ads_displayed"] += 1
            elif interaction["event_type"] == "BuyPremium":
                pass
            elif interaction["event_type"] in ("Play", "Like", "Skip"):
                if interaction["track_id"] != old["track_id"] or interaction["user_id"] != old["user_id"]:
                    # this attribute considers play/like/skip of the same track as one track listened
                    user["tracks_listened"] += 1
                process_track_interaction(interaction, user, tracks, artists, genres)
            else:
                print(f"Unknown interaction: {interaction['event_type']}")

    user["time_spent_s"] += (to_datetime(interaction["timestamp"]) - session_start).seconds
    user["sessions"] += 1

    max_days = (to_datetime(interaction["timestamp"]) - first_date).days
    return max_days

def postprocess_users(users: dict, max_days: int):
    for key in users.keys():
        user = users[key]
        user["account_age"] = max_days - user["account_age"]
        user["genre_preferences"] = [p / max(1, user["total_weight"]) for p in user["genre_preferences"]]
        user["track_preferences"] = [p / max(1, user["total_weight"]) for p in user["track_preferences"]]
        user["time_preferences"] = [p / max(1, user["plays"]) for p in user["time_preferences"]]

        user["sessions_daily"] = user["sessions"] / max(1, user["account_age"])
        user["time_spent_daily_s"] = user["time_spent_s"] / max(1, user["account_age"])
        user["time_spent_per_session_s"] = user["time_spent_s"] / max(1, user["sessions"])
        user["ads_watch_rate"] = user["ads_watched"] / max(1, user["ads_displayed"])

def serialize(user: dict) -> str:
    output = f"{user['premium']} {user['sex']} {user['salary']} {user['account_age']} {user['sessions']} " + \
             f"{user['sessions_daily']} {user['time_spent_s']} {user['time_spent_daily_s']} " + \
             f"{user['time_spent_per_session_s']} {user['ads_displayed']} {user['ads_watched']} " + \
             f"{user['ads_watch_rate']} {user['total_weight']} {user['tracks_listened']} {user['plays']} " + \
             f"{user['likes']} {user['skips']}"
    for time in user['time_preferences']:
        output += f" {time}"
    for track in user['track_preferences']:
        output += f" {track}"
    for genre in user['genre_preferences']:
        output += f" {genre}"
    output += "\n"
    return output


def genre_cutoff(first_genre_attr):
    x, y, genres2 = [], [], []
    with open("training.txt", "r") as file:
        for line in file:
            data = [float(item) for item in line.split()]
            x.append(data[first_genre_attr:])
            y.append(data[0])

    nonexistent, irrelevant, count = 0, 0, 0
    for i in range(len(TRACK_NONZERO_GENRES)):
        values = [c[i] for c in x]
        if sum(values) < 1:
            nonexistent += sum(values)
            continue
        if abs(np.corrcoef(values, y)[0, 1]) < 0.3:
            irrelevant += sum(values)
            continue
        genres2.append(TRACK_NONZERO_GENRES[i])
        print(TRACK_NONZERO_GENRES[i], np.corrcoef(values, y)[0, 1], sum(values))
        count += 1

    print(nonexistent)
    print(irrelevant)
    print(len(x[0]))
    print(count)
    print(genres2)

if __name__ == "__main__":
    pass
