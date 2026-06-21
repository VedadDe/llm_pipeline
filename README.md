# Titanic AI Agent

Jednostavna konzolna aplikacija koja prima instrukciju
i pretvara je u validiran plan treniranja modela masinskog ucenja, trenira
binarni klasifikator za predikciju kolone `Survived` i ispisuje izvjestaj sa
confusion matrix, Accuracy, Precision, Recall i F1 metrikama.

## Struktura projekta

`main.py` - CLI entry point, pokrece se sa `python -m main`
`llm_parser.py` - parsiranje instrukcije, OpenRouter integracija i validacija plana
`ml_pipeline.py` - pandas i scikit-learn pipeline za treniranje modela
`reporting.py` - formatiranje i snimanje izvjestaja
`data/titanic.csv` - podrazumijevani Titanic dataset
`examples/` - primjeri instrukcija za brzo testiranje
`requirements.txt` - Python dependencies
`.env.example` - primjer lokalne konfiguracije za API kljuc
`questions.txt` - odgovori na dodatna pitanja iz zadatka

poitrebno je napravit .env po template u .env.example 

## Instalacija

Preporucena verzija je Python 3.10 ili novija.

Na Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Na macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Konfiguracija API key

OpenRouter je LLM provider za parsiranje instrukcija, ukoliko nije konfigurisan radice fallback opcija. Treba napomenuti da fallback opcija 
nece kvalitetno parsirati i evaluirati typo greske sto je moze ciniti neefikasnom za zadatak, ali ipak ce ciniti da aplikacija radi (uradjeno da CLI ostane testabilan)

Napraviti `.env`
fajl u root direktoriju projekta:

```env
OPENROUTER_API_KEY=your_real_key_here
OPENROUTER_MODEL=cohere/north-mini-code:free
```

## Pokretanje


Pokretanje sa instrukcijom direktno iz terminala:

```powershell
python -m main --instruction "Train a random forest model with features: Pclass, Sex, Age, Fare"
```

Pokretanje sa instrukcijom iz fajla:

```powershell
python -m main --infile examples\age_xgboost.txt
```

Snimanje izvjestaja u fajl:

```powershell
python -m main --infile examples\age_xgboost.txt --outfile results.txt
```

Koristenje drugog CSV fajla sa istim datasetom se moze uradti na nacin (vazno naglasiti istim datasetom):

```powershell
python -m main --instruction "Train with Age, Sex and Fare" --data data\titanic.csv
```

## Primjeri instrukcija

```text
Train with features: Age, Sex and Fare using avg imputation, scaling, and test size 25%
Train a random forest model with features: Pclass, Sex, Age, Fare
```

## Izlaz

CLI ispisuje:

originalnu instrukciju
validirani plan
koristeni model
koristene feature kolone
broj redova u train i test skupu
confusion matrix
Accuracy, Precision, Recall i F1
upozorenja, ako je nesto fallback-ano ili ignorisano

Primjer ispisa:
 python -m main --instruction "Drop all columns except Age, and train a xgboost model"
Report

Instruction:
Drop all columns except Age, and train a xgboost model

Resolved plan:
- raw_instruction: Drop all columns except Age, and train a xgboost model
- target: Survived
- selected_features: ['Age']
- drop_features: []
- model: xgboost
- numeric_imputation: median
- categorical_imputation: most_frequent
- scaling: False
- test_size: 0.2
- parser: openrouter

Status: trained
Data path: data\titanic.csv
Rows: 891
Train rows: 712
Test rows: 179
Model used: xgboost
Features used: ['Age']

Metrics:
Confusion matrix: [[90, 20], [55, 14]]
Accuracy: 0.5810
Precision: 0.4118
Recall: 0.2029
F1: 0.2718

Warnings:
- None



Vazno je napomenuti da aplikacija nece prihvatiti nepoznat model kao da je
validan zahtjev. Ako korisnik trazi nesto izvan definicije koda, aplikacija ce koristiti default model ali rezultat ce biti objasnjen.

Podrzani modeli

`logistic_regression`
`decision_tree`
`random_forest`
`gradient_boosting`
`xgboost`

Podrzane numericke imputacije:

`mean`
`median`
`most_frequent`

Podrzane kategoricke imputacije:

`most_frequent`
`constant`

`xgboost` je ukljucen u `requirements.txt`. Ako nije instaliran, pipeline koristi
 `gradient_boosting` i ispisuje upozorenje.

## Objasnjenje arhitekture

Instrukcija korisnika -> LLM/parser kreira plan -> postojeci Python code trenira i evaluira

Rjesenje nije dizajnirano da dopusti llm- u da kreira python kod za treniranje modela jer to sa aspekta sigurnosti nije najbolja opcija obzirom da llm moze generisati "svasta" (prompt injection).
Rjesenje je implementirano tako da postoji ml_pipeline kojem se proslijedjuje plan izvrsavanja (koji parametri se podesavaju u pipeline- u i koji model se koristi).
Potom se izvrsavaju unaprijed definisani koraci, a rezultati se ispisuju ili spasavaju. 

Korisnik instrukcijama definise:

- koje karakteristike odbaciti ili koristiti
- koji model od ponudjenih trenirati
- preprocesiranje
- train/test split
- evaluacijski ispis/ spasavanje

Rjesenje je napravljeno modularno, i to>

- `main.py` je zaduzen samo za CLI ulaz i orkestraciju
- `llm_parser.py` odvaja LLM parsiranje i vraca plan izvrsavanja
- `ml_pipeline.py` sadrzi ML logiku i ne zavisi od toga da li je plan dosao od LLM-a ili fallback parsera
- `reporting.py` izoluje formatiranje izlaza.

Ovako dizajnirano rjesenje cini da greska u jednom dijelu bude izolovana i ne odrazava se direktno na ostale (relativno nezavisne) dijelove. 

LLM se koristi samo za parsiranje i razumijevanje korisnickih uputa a ne za izvrsavanjekoda. 

Rezultujuca kolona `Survived` se eksplicitno odbija kao input feature.

## ogranicenja

- Zadatak je fiksiran na binarnu klasifikaciju kolone `Survived`
- Parser podrzava ogranicen skup modela i preprocessing opcija
- Fallback parser je namijenjen jednostavnim (ispravno napisanim i jasno definisanim) instrukcijama, ne opstem NLP-u i razumjevanju teksta 


# Odgovori

## 1

Podrzavanje bilo kog skupa podataka. 

Potrebno je napravit konfigurabilan prompt, koji bi se cuvao vjerovatno u bazi i bio lahko izmjenljiv a ne hard-kodiran u kodu kao sto je u kodu trenutno. Cuvao bih ga u .json formatu unutar sql baze. Dodao 
bih i odvojena kreiranja scheme (koja bih na isti nacin cuvao u bazi) sa informacijama o nazivima kolona, tipovima kolona, dozvoljenim modelima, pretprocesiranju i itd. Dakle svaki data-set bi imao posebnu i odvojenu domensku konfiguraciju.
Cuvanje u bazi omogucava lahku izmjenu  bez deploya aplikacije.
Aplikacija prvo ucitava konfiguraciju, zatim LLM-u salje odvojen prompt zajedno sa odgovarajucom schemom kako bi llm vratio plan samo iz dozvoljenih opcija. Nakon toga se pipeline izvrsava kao i do sad (uz naravno izmjene i konfiguraciju dodatnih modela). Citav dio koda vezan za dodavanje modela moze biti konfigurabilan i sacuvan externo u bazi ( u vidu skripti) kako bi se bez deploya novi modeli mogli dodavati (ako su instalirani naravno). Ovo omogucava izvojeno pokretanje koda. Sijedeci ovaj pristup pravi se siguran sistem, koji je i dalje kontrolisan. 

## 2

Podrzavanje opcije chata umjesto jedne instrukcije

Umjesto jedne instrukcije, dodao bih opciju sesije na nacin da spasavam razgovor - trenutno stanje korsnickih zahtjeva (npr. plan iz llma, dataset koji se koristi, model i itd) u state ili bazi. 
Svaka nova poruka prema LLM- u (svaki prompt) bi azurirala potrebne informacije u state- u/ bazi. Trening bi se pokrenuo tek nakon korisnicke potvrde, sa parametrima dobijenim iz plana na osnovu posljednjih informacija spasenih u bazi pomocu ranije opisanim azuriranjem. Trening se obavlja potom prolaskom kroz pipeline sa propagiranjem plana iz prethodno opisanog koraka. 

## 3
EDA Features

Ovo bih uradio pomocu koraka pretprocesiranja podataka- dataseta. Ciljao bih informacije koje mogu bit znacajne za deskriptivne statistike kao sto su tipovi kolona, distribucije, srednje vrijednosti, range, min-max, frekvencije i dr. Sve bih ih spasio u bazu kao opis dataseta i to na naredni nacin. Kako datasetovi mogu biti razliciti, sa razlicitim kolonama (kategoricke, ordered, numerical, itd), informacije bih spasio u obliku .json- a. Ovim mozemo racunati deskriptivne statistike za svaki skup podataka i to za svaku pojedinacnu kolonu dataseta (jer .json dozvoljava nesting). Korisnik 
kreira upit prema LLM- u koji vrsi kreiranje plana (slicno kao u trenutnom kodu samo malo jednostavnije). Plan se proslijedjuje .py skripti koja iz baze vadi trazene informacije na osnovu propagiranog plana (ukoliko su dostupne) i potom ih vraca natrag za ispis. 
Takodjer po zaprimanju novog dataseta, uz izracunavanje svih deskriptivnih statistika poslao bih paralelno llmu zahtjev za prijednlog korisnih transformacija i kreiranje plana preprocesuiranja dataseta. To moze biti npr: izbacivanje nekvalitetnih kolona, kreiranje novih feature-a spojem drugih featurea (feature engineering) i dr. Ipak, ovdje moramo biti pazljivi da korisnik mora potvrditi sve izmjene (kao sto je to slucaj kod poznatih agent alata danas npr claude ili codex). Za sve predlozene izmjene treba postojat .py skripta, ili opcija da se ona embeda u sistem externo. Postoji nacin da se skripta ubaci u bazu, kreira executor za skripte koji se potom pozove kad se odredjena skripta zatrazi. One opracije za koje ne postoji skripta ne mogu biti ni izvrsne (isto kao sto vrijedi u trenutnoj implmentaciji za ml trening). Nakon sto korisnik potvrdi, deskriptivne statistike se prosiruju za novo izracunate. 

Naravno ovo je za slucaj da zelimo dinamicki dodavat bez novog deploy- a. Ukoliko nije problem raditi deploy koda zbog dodavanja - prosirivanja novim funkcionalnostima to je bolja opcija 