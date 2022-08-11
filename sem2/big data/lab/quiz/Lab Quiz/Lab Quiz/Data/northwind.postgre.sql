--
-- postgresql northwind database v1.0 from ramiro estigarribia canese  
-- you may contact him at email   ramiro.estigarribia@rieder.com.py 
--


set statement_timeout = 0;
set client_encoding = 'utf8';
set standard_conforming_strings = on;
set check_function_bodies = false;
set client_min_messages = warning;

--
-- name: plpgsql; type: extension; schema: -; owner: 
--

create extension if not exists plpgsql with schema pg_catalog;


--
-- name: extension plpgsql; type: comment; schema: -; owner: 
--

comment on extension plpgsql is 'pl/pgsql procedural language';


set search_path = public, pg_catalog;

set default_tablespace = '';

set default_with_oids = false;

--
-- name: categories; type: table; schema: public; owner: postgres; tablespace: 
--

create table categories (
    "categoryid" smallint not null,
    "categoryname" character varying(15) not null,
    "description" text,
    "picture" bytea
);


alter table public.categories owner to postgres;

--
-- name: customercustomerdemo; type: table; schema: public; owner: postgres; tablespace: 
--

create table customercustomerdemo (
    "customerid" bpchar not null,
    "customertypeid" bpchar not null
);


alter table public.customercustomerdemo owner to postgres;

--
-- name: customerdemographics; type: table; schema: public; owner: postgres; tablespace: 
--

create table customerdemographics (
    "customertypeid" bpchar not null,
    "customerdesc" text
);


alter table public.customerdemographics owner to postgres;

--
-- name: customers; type: table; schema: public; owner: postgres; tablespace: 
--

create table customers (
    "customerid" bpchar not null,
    "companyname" character varying(40) not null,
    "contactname" character varying(30),
    "contacttitle" character varying(30),
    "address" character varying(60),
    "city" character varying(15),
    "region" character varying(15),
    "postalcode" character varying(10),
    "country" character varying(15),
    "phone" character varying(24),
    "fax" character varying(24)
);


alter table public.customers owner to postgres;

--
-- name: employees; type: table; schema: public; owner: postgres; tablespace: 
--

create table employees (
    "employeeid" smallint not null,
    "lastname" character varying(20) not null,
    "firstname" character varying(10) not null,
    "title" character varying(30),
    "titleofcourtesy" character varying(25),
    "birthdate" date,
    "hiredate" date,
    "address" character varying(60),
    "city" character varying(15),
    "region" character varying(15),
    "postalcode" character varying(10),
    "country" character varying(15),
    "homephone" character varying(24),
    "extension" character varying(4),
    "photo" bytea,
    "notes" text,
    "reportsto" smallint,
    "photopath" character varying(255)
);


alter table public.employees owner to postgres;

--
-- name: employeeterritories; type: table; schema: public; owner: postgres; tablespace: 
--

create table employeeterritories (
    "employeeid" smallint not null,
    "territoryid" character varying(20) not null
);


alter table public.employeeterritories owner to postgres;

--
-- name: order_details; type: table; schema: public; owner: postgres; tablespace: 
--

create table order_details (
    "orderid" smallint not null,
    "productid" smallint not null,
    "unitprice" real not null,
    "quantity" smallint not null,
    "discount" real not null
);


alter table public.order_details owner to postgres;

--
-- name: orders; type: table; schema: public; owner: postgres; tablespace: 
--

create table orders (
    "orderid" smallint not null,
    "customerid" bpchar,
    "employeeid" smallint,
    "orderdate" date,
    "requireddate" date,
    "shippeddate" date,
    "shipvia" smallint,
    "freight" real,
    "shipname" character varying(40),
    "shipaddress" character varying(60),
    "shipcity" character varying(15),
    "shipregion" character varying(15),
    "shippostalcode" character varying(10),
    "shipcountry" character varying(15)
);


alter table public.orders owner to postgres;

--
-- name: products; type: table; schema: public; owner: postgres; tablespace: 
--

create table products (
    "productid" smallint not null,
    "productname" character varying(40) not null,
    "supplierid" smallint,
    "categoryid" smallint,
    "quantityperunit" character varying(20),
    "unitprice" real,
    "unitsinstock" smallint,
    "unitsonorder" smallint,
    "reorderlevel" smallint,
    "discontinued" integer not null
);


alter table public.products owner to postgres;

--
-- name: region; type: table; schema: public; owner: postgres; tablespace: 
--

create table region (
    "regionid" smallint not null,
    "regiondescription" bpchar not null
);


alter table public.region owner to postgres;

--
-- name: shippers; type: table; schema: public; owner: postgres; tablespace: 
--

create table shippers (
    "shipperid" smallint not null,
    "companyname" character varying(40) not null,
    "phone" character varying(24)
);


alter table public.shippers owner to postgres;

--
-- name: shippers_tmp; type: table; schema: public; owner: postgres; tablespace: 
--

create table shippers_tmp (
    "shipperid" smallint not null,
    "companyname" character varying(40) not null,
    "phone" character varying(24)
);


alter table public.shippers_tmp owner to postgres;

--
-- name: suppliers; type: table; schema: public; owner: postgres; tablespace: 
--

create table suppliers (
    "supplierid" smallint not null,
    "companyname" character varying(40) not null,
    "contactname" character varying(30),
    "contacttitle" character varying(30),
    "address" character varying(60),
    "city" character varying(15),
    "region" character varying(15),
    "postalcode" character varying(10),
    "country" character varying(15),
    "phone" character varying(24),
    "fax" character varying(24),
    "homepage" text
);


alter table public.suppliers owner to postgres;

--
-- name: territories; type: table; schema: public; owner: postgres; tablespace: 
--

create table territories (
    "territoryid" character varying(20) not null,
    "territorydescription" bpchar not null,
    "regionid" smallint not null
);


alter table public.territories owner to postgres;

--
-- name: usstates; type: table; schema: public; owner: postgres; tablespace: 
--

create table usstates (
    "stateid" smallint not null,
    "statename" character varying(100),
    "stateabbr" character varying(2),
    "stateregion" character varying(50)
);


alter table public.usstates owner to postgres;

--
-- data for name: categories; type: table data; schema: public; owner: postgres
--

insert into categories values (1, 'beverages', 'soft drinks, coffees, teas, beers, and ales', '\x');
insert into categories values (2, 'condiments', 'sweet and savory sauces, relishes, spreads, and seasonings', '\x');
insert into categories values (3, 'confections', 'desserts, candies, and sweet breads', '\x');
insert into categories values (4, 'dairy products', 'cheeses', '\x');
insert into categories values (5, 'grains/cereals', 'breads, crackers, pasta, and cereal', '\x');
insert into categories values (6, 'meat/poultry', 'prepared meats', '\x');
insert into categories values (7, 'produce', 'dried fruit and bean curd', '\x');
insert into categories values (8, 'seafood', 'seaweed and fish', '\x');


--
-- data for name: customercustomerdemo; type: table data; schema: public; owner: postgres
--



--
-- data for name: customerdemographics; type: table data; schema: public; owner: postgres
--



--
-- data for name: customers; type: table data; schema: public; owner: postgres
--

insert into customers values ('alfki', 'alfreds futterkiste', 'maria anders', 'sales representative', 'obere str. 57', 'berlin', null, '12209', 'germany', '030-0074321', '030-0076545');
insert into customers values ('anatr', 'ana trujillo emparedados y helados', 'ana trujillo', 'owner', 'avda. de la constitución 2222', 'méxico d.f.', null, '05021', 'mexico', '(5) 555-4729', '(5) 555-3745');
insert into customers values ('anton', 'antonio moreno taquería', 'antonio moreno', 'owner', 'mataderos  2312', 'méxico d.f.', null, '05023', 'mexico', '(5) 555-3932', null);
insert into customers values ('arout', 'around the horn', 'thomas hardy', 'sales representative', '120 hanover sq.', 'london', null, 'wa1 1dp', 'uk', '(171) 555-7788', '(171) 555-6750');
insert into customers values ('bergs', 'berglunds snabbköp', 'christina berglund', 'order administrator', 'berguvsvägen  8', 'luleå', null, 's-958 22', 'sweden', '0921-12 34 65', '0921-12 34 67');
insert into customers values ('blaus', 'blauer see delikatessen', 'hanna moos', 'sales representative', 'forsterstr. 57', 'mannheim', null, '68306', 'germany', '0621-08460', '0621-08924');
insert into customers values ('blonp', 'blondesddsl père et fils', 'frédérique citeaux', 'marketing manager', '24, place kléber', 'strasbourg', null, '67000', 'france', '88.60.15.31', '88.60.15.32');
insert into customers values ('bolid', 'bólido comidas preparadas', 'martín sommer', 'owner', 'c/ araquil, 67', 'madrid', null, '28023', 'spain', '(91) 555 22 82', '(91) 555 91 99');
insert into customers values ('bonap', 'bon app''', 'laurence lebihan', 'owner', '12, rue des bouchers', 'marseille', null, '13008', 'france', '91.24.45.40', '91.24.45.41');
insert into customers values ('bottm', 'bottom-dollar markets', 'elizabeth lincoln', 'accounting manager', '23 tsawassen blvd.', 'tsawassen', 'bc', 't2f 8m4', 'canada', '(604) 555-4729', '(604) 555-3745');
insert into customers values ('bsbev', 'b''s beverages', 'victoria ashworth', 'sales representative', 'fauntleroy circus', 'london', null, 'ec2 5nt', 'uk', '(171) 555-1212', null);
insert into customers values ('cactu', 'cactus comidas para llevar', 'patricio simpson', 'sales agent', 'cerrito 333', 'buenos aires', null, '1010', 'argentina', '(1) 135-5555', '(1) 135-4892');
insert into customers values ('centc', 'centro comercial moctezuma', 'francisco chang', 'marketing manager', 'sierras de granada 9993', 'méxico d.f.', null, '05022', 'mexico', '(5) 555-3392', '(5) 555-7293');
insert into customers values ('chops', 'chop-suey chinese', 'yang wang', 'owner', 'hauptstr. 29', 'bern', null, '3012', 'switzerland', '0452-076545', null);
insert into customers values ('commi', 'comércio mineiro', 'pedro afonso', 'sales associate', 'av. dos lusíadas, 23', 'sao paulo', 'sp', '05432-043', 'brazil', '(11) 555-7647', null);
insert into customers values ('consh', 'consolidated holdings', 'elizabeth brown', 'sales representative', 'berkeley gardens 12  brewery', 'london', null, 'wx1 6lt', 'uk', '(171) 555-2282', '(171) 555-9199');
insert into customers values ('dracd', 'drachenblut delikatessen', 'sven ottlieb', 'order administrator', 'walserweg 21', 'aachen', null, '52066', 'germany', '0241-039123', '0241-059428');
insert into customers values ('dumon', 'du monde entier', 'janine labrune', 'owner', '67, rue des cinquante otages', 'nantes', null, '44000', 'france', '40.67.88.88', '40.67.89.89');
insert into customers values ('eastc', 'eastern connection', 'ann devon', 'sales agent', '35 king george', 'london', null, 'wx3 6fw', 'uk', '(171) 555-0297', '(171) 555-3373');
insert into customers values ('ernsh', 'ernst handel', 'roland mendel', 'sales manager', 'kirchgasse 6', 'graz', null, '8010', 'austria', '7675-3425', '7675-3426');
insert into customers values ('famia', 'familia arquibaldo', 'aria cruz', 'marketing assistant', 'rua orós, 92', 'sao paulo', 'sp', '05442-030', 'brazil', '(11) 555-9857', null);
insert into customers values ('fissa', 'fissa fabrica inter. salchichas s.a.', 'diego roel', 'accounting manager', 'c/ moralzarzal, 86', 'madrid', null, '28034', 'spain', '(91) 555 94 44', '(91) 555 55 93');
insert into customers values ('folig', 'folies gourmandes', 'martine rancé', 'assistant sales agent', '184, chaussée de tournai', 'lille', null, '59000', 'france', '20.16.10.16', '20.16.10.17');
insert into customers values ('folko', 'folk och fä hb', 'maria larsson', 'owner', 'åkergatan 24', 'bräcke', null, 's-844 67', 'sweden', '0695-34 67 21', null);
insert into customers values ('frank', 'frankenversand', 'peter franken', 'marketing manager', 'berliner platz 43', 'münchen', null, '80805', 'germany', '089-0877310', '089-0877451');
insert into customers values ('franr', 'france restauration', 'carine schmitt', 'marketing manager', '54, rue royale', 'nantes', null, '44000', 'france', '40.32.21.21', '40.32.21.20');
insert into customers values ('frans', 'franchi s.p.a.', 'paolo accorti', 'sales representative', 'via monte bianco 34', 'torino', null, '10100', 'italy', '011-4988260', '011-4988261');
insert into customers values ('furib', 'furia bacalhau e frutos do mar', 'lino rodriguez', 'sales manager', 'jardim das rosas n. 32', 'lisboa', null, '1675', 'portugal', '(1) 354-2534', '(1) 354-2535');
insert into customers values ('galed', 'galería del gastrónomo', 'eduardo saavedra', 'marketing manager', 'rambla de cataluña, 23', 'barcelona', null, '08022', 'spain', '(93) 203 4560', '(93) 203 4561');
insert into customers values ('godos', 'godos cocina típica', 'josé pedro freyre', 'sales manager', 'c/ romero, 33', 'sevilla', null, '41101', 'spain', '(95) 555 82 82', null);
insert into customers values ('gourl', 'gourmet lanchonetes', 'andré fonseca', 'sales associate', 'av. brasil, 442', 'campinas', 'sp', '04876-786', 'brazil', '(11) 555-9482', null);
insert into customers values ('greal', 'great lakes food market', 'howard snyder', 'marketing manager', '2732 baker blvd.', 'eugene', 'or', '97403', 'usa', '(503) 555-7555', null);
insert into customers values ('grosr', 'grosella-restaurante', 'manuel pereira', 'owner', '5ª ave. los palos grandes', 'caracas', 'df', '1081', 'venezuela', '(2) 283-2951', '(2) 283-3397');
insert into customers values ('hanar', 'hanari carnes', 'mario pontes', 'accounting manager', 'rua do paço, 67', 'rio de janeiro', 'rj', '05454-876', 'brazil', '(21) 555-0091', '(21) 555-8765');
insert into customers values ('hilaa', 'hilarion-abastos', 'carlos hernández', 'sales representative', 'carrera 22 con ave. carlos soublette #8-35', 'san cristóbal', 'táchira', '5022', 'venezuela', '(5) 555-1340', '(5) 555-1948');
insert into customers values ('hungc', 'hungry coyote import store', 'yoshi latimer', 'sales representative', 'city center plaza 516 main st.', 'elgin', 'or', '97827', 'usa', '(503) 555-6874', '(503) 555-2376');
insert into customers values ('hungo', 'hungry owl all-night grocers', 'patricia mckenna', 'sales associate', '8 johnstown road', 'cork', 'co. cork', null, 'ireland', '2967 542', '2967 3333');
insert into customers values ('islat', 'island trading', 'helen bennett', 'marketing manager', 'garden house crowther way', 'cowes', 'isle of wight', 'po31 7pj', 'uk', '(198) 555-8888', null);
insert into customers values ('koene', 'königlich essen', 'philip cramer', 'sales associate', 'maubelstr. 90', 'brandenburg', null, '14776', 'germany', '0555-09876', null);
insert into customers values ('lacor', 'la corne d''abondance', 'daniel tonini', 'sales representative', '67, avenue de l''europe', 'versailles', null, '78000', 'france', '30.59.84.10', '30.59.85.11');
insert into customers values ('lamai', 'la maison d''asie', 'annette roulet', 'sales manager', '1 rue alsace-lorraine', 'toulouse', null, '31000', 'france', '61.77.61.10', '61.77.61.11');
insert into customers values ('laugb', 'laughing bacchus wine cellars', 'yoshi tannamuri', 'marketing assistant', '1900 oak st.', 'vancouver', 'bc', 'v3f 2k1', 'canada', '(604) 555-3392', '(604) 555-7293');
insert into customers values ('lazyk', 'lazy k kountry store', 'john steel', 'marketing manager', '12 orchestra terrace', 'walla walla', 'wa', '99362', 'usa', '(509) 555-7969', '(509) 555-6221');
insert into customers values ('lehms', 'lehmanns marktstand', 'renate messner', 'sales representative', 'magazinweg 7', 'frankfurt a.m.', null, '60528', 'germany', '069-0245984', '069-0245874');
insert into customers values ('letss', 'let''s stop n shop', 'jaime yorres', 'owner', '87 polk st. suite 5', 'san francisco', 'ca', '94117', 'usa', '(415) 555-5938', null);
insert into customers values ('lilas', 'lila-supermercado', 'carlos gonzález', 'accounting manager', 'carrera 52 con ave. bolívar #65-98 llano largo', 'barquisimeto', 'lara', '3508', 'venezuela', '(9) 331-6954', '(9) 331-7256');
insert into customers values ('linod', 'lino-delicateses', 'felipe izquierdo', 'owner', 'ave. 5 de mayo porlamar', 'i. de margarita', 'nueva esparta', '4980', 'venezuela', '(8) 34-56-12', '(8) 34-93-93');
insert into customers values ('lonep', 'lonesome pine restaurant', 'fran wilson', 'sales manager', '89 chiaroscuro rd.', 'portland', 'or', '97219', 'usa', '(503) 555-9573', '(503) 555-9646');
insert into customers values ('magaa', 'magazzini alimentari riuniti', 'giovanni rovelli', 'marketing manager', 'via ludovico il moro 22', 'bergamo', null, '24100', 'italy', '035-640230', '035-640231');
insert into customers values ('maisd', 'maison dewey', 'catherine dewey', 'sales agent', 'rue joseph-bens 532', 'bruxelles', null, 'b-1180', 'belgium', '(02) 201 24 67', '(02) 201 24 68');
insert into customers values ('merep', 'mère paillarde', 'jean fresnière', 'marketing assistant', '43 rue st. laurent', 'montréal', 'québec', 'h1j 1c3', 'canada', '(514) 555-8054', '(514) 555-8055');
insert into customers values ('morgk', 'morgenstern gesundkost', 'alexander feuer', 'marketing assistant', 'heerstr. 22', 'leipzig', null, '04179', 'germany', '0342-023176', null);
insert into customers values ('norts', 'north/south', 'simon crowther', 'sales associate', 'south house 300 queensbridge', 'london', null, 'sw7 1rz', 'uk', '(171) 555-7733', '(171) 555-2530');
insert into customers values ('ocean', 'océano atlántico ltda.', 'yvonne moncada', 'sales agent', 'ing. gustavo moncada 8585 piso 20-a', 'buenos aires', null, '1010', 'argentina', '(1) 135-5333', '(1) 135-5535');
insert into customers values ('oldwo', 'old world delicatessen', 'rene phillips', 'sales representative', '2743 bering st.', 'anchorage', 'ak', '99508', 'usa', '(907) 555-7584', '(907) 555-2880');
insert into customers values ('ottik', 'ottilies käseladen', 'henriette pfalzheim', 'owner', 'mehrheimerstr. 369', 'köln', null, '50739', 'germany', '0221-0644327', '0221-0765721');
insert into customers values ('paris', 'paris spécialités', 'marie bertrand', 'owner', '265, boulevard charonne', 'paris', null, '75012', 'france', '(1) 42.34.22.66', '(1) 42.34.22.77');
insert into customers values ('peric', 'pericles comidas clásicas', 'guillermo fernández', 'sales representative', 'calle dr. jorge cash 321', 'méxico d.f.', null, '05033', 'mexico', '(5) 552-3745', '(5) 545-3745');
insert into customers values ('picco', 'piccolo und mehr', 'georg pipps', 'sales manager', 'geislweg 14', 'salzburg', null, '5020', 'austria', '6562-9722', '6562-9723');
insert into customers values ('prini', 'princesa isabel vinhos', 'isabel de castro', 'sales representative', 'estrada da saúde n. 58', 'lisboa', null, '1756', 'portugal', '(1) 356-5634', null);
insert into customers values ('quede', 'que delícia', 'bernardo batista', 'accounting manager', 'rua da panificadora, 12', 'rio de janeiro', 'rj', '02389-673', 'brazil', '(21) 555-4252', '(21) 555-4545');
insert into customers values ('queen', 'queen cozinha', 'lúcia carvalho', 'marketing assistant', 'alameda dos canàrios, 891', 'sao paulo', 'sp', '05487-020', 'brazil', '(11) 555-1189', null);
insert into customers values ('quick', 'quick-stop', 'horst kloss', 'accounting manager', 'taucherstraße 10', 'cunewalde', null, '01307', 'germany', '0372-035188', null);
insert into customers values ('ranch', 'rancho grande', 'sergio gutiérrez', 'sales representative', 'av. del libertador 900', 'buenos aires', null, '1010', 'argentina', '(1) 123-5555', '(1) 123-5556');
insert into customers values ('rattc', 'rattlesnake canyon grocery', 'paula wilson', 'assistant sales representative', '2817 milton dr.', 'albuquerque', 'nm', '87110', 'usa', '(505) 555-5939', '(505) 555-3620');
insert into customers values ('reggc', 'reggiani caseifici', 'maurizio moroni', 'sales associate', 'strada provinciale 124', 'reggio emilia', null, '42100', 'italy', '0522-556721', '0522-556722');
insert into customers values ('ricar', 'ricardo adocicados', 'janete limeira', 'assistant sales agent', 'av. copacabana, 267', 'rio de janeiro', 'rj', '02389-890', 'brazil', '(21) 555-3412', null);
insert into customers values ('ricsu', 'richter supermarkt', 'michael holz', 'sales manager', 'grenzacherweg 237', 'genève', null, '1203', 'switzerland', '0897-034214', null);
insert into customers values ('romey', 'romero y tomillo', 'alejandra camino', 'accounting manager', 'gran vía, 1', 'madrid', null, '28001', 'spain', '(91) 745 6200', '(91) 745 6210');
insert into customers values ('santg', 'santé gourmet', 'jonas bergulfsen', 'owner', 'erling skakkes gate 78', 'stavern', null, '4110', 'norway', '07-98 92 35', '07-98 92 47');
insert into customers values ('savea', 'save-a-lot markets', 'jose pavarotti', 'sales representative', '187 suffolk ln.', 'boise', 'id', '83720', 'usa', '(208) 555-8097', null);
insert into customers values ('seves', 'seven seas imports', 'hari kumar', 'sales manager', '90 wadhurst rd.', 'london', null, 'ox15 4nb', 'uk', '(171) 555-1717', '(171) 555-5646');
insert into customers values ('simob', 'simons bistro', 'jytte petersen', 'owner', 'vinbæltet 34', 'kobenhavn', null, '1734', 'denmark', '31 12 34 56', '31 13 35 57');
insert into customers values ('specd', 'spécialités du monde', 'dominique perrier', 'marketing manager', '25, rue lauriston', 'paris', null, '75016', 'france', '(1) 47.55.60.10', '(1) 47.55.60.20');
insert into customers values ('splir', 'split rail beer & ale', 'art braunschweiger', 'sales manager', 'p.o. box 555', 'lander', 'wy', '82520', 'usa', '(307) 555-4680', '(307) 555-6525');
insert into customers values ('suprd', 'suprêmes délices', 'pascale cartrain', 'accounting manager', 'boulevard tirou, 255', 'charleroi', null, 'b-6000', 'belgium', '(071) 23 67 22 20', '(071) 23 67 22 21');
insert into customers values ('thebi', 'the big cheese', 'liz nixon', 'marketing manager', '89 jefferson way suite 2', 'portland', 'or', '97201', 'usa', '(503) 555-3612', null);
insert into customers values ('thecr', 'the cracker box', 'liu wong', 'marketing assistant', '55 grizzly peak rd.', 'butte', 'mt', '59801', 'usa', '(406) 555-5834', '(406) 555-8083');
insert into customers values ('tomsp', 'toms spezialitäten', 'karin josephs', 'marketing manager', 'luisenstr. 48', 'münster', null, '44087', 'germany', '0251-031259', '0251-035695');
insert into customers values ('tortu', 'tortuga restaurante', 'miguel angel paolino', 'owner', 'avda. azteca 123', 'méxico d.f.', null, '05033', 'mexico', '(5) 555-2933', null);
insert into customers values ('tradh', 'tradição hipermercados', 'anabela domingues', 'sales representative', 'av. inês de castro, 414', 'sao paulo', 'sp', '05634-030', 'brazil', '(11) 555-2167', '(11) 555-2168');
insert into customers values ('traih', 'trail''s head gourmet provisioners', 'helvetius nagy', 'sales associate', '722 davinci blvd.', 'kirkland', 'wa', '98034', 'usa', '(206) 555-8257', '(206) 555-2174');
insert into customers values ('vaffe', 'vaffeljernet', 'palle ibsen', 'sales manager', 'smagsloget 45', 'århus', null, '8200', 'denmark', '86 21 32 43', '86 22 33 44');
insert into customers values ('victe', 'victuailles en stock', 'mary saveley', 'sales agent', '2, rue du commerce', 'lyon', null, '69004', 'france', '78.32.54.86', '78.32.54.87');
insert into customers values ('vinet', 'vins et alcools chevalier', 'paul henriot', 'accounting manager', '59 rue de l''abbaye', 'reims', null, '51100', 'france', '26.47.15.10', '26.47.15.11');
insert into customers values ('wandk', 'die wandernde kuh', 'rita müller', 'sales representative', 'adenauerallee 900', 'stuttgart', null, '70563', 'germany', '0711-020361', '0711-035428');
insert into customers values ('warth', 'wartian herkku', 'pirkko koskitalo', 'accounting manager', 'torikatu 38', 'oulu', null, '90110', 'finland', '981-443655', '981-443655');
insert into customers values ('welli', 'wellington importadora', 'paula parente', 'sales manager', 'rua do mercado, 12', 'resende', 'sp', '08737-363', 'brazil', '(14) 555-8122', null);
insert into customers values ('whitc', 'white clover markets', 'karl jablonski', 'owner', '305 - 14th ave. s. suite 3b', 'seattle', 'wa', '98128', 'usa', '(206) 555-4112', '(206) 555-4115');
insert into customers values ('wilmk', 'wilman kala', 'matti karttunen', 'owner/marketing assistant', 'keskuskatu 45', 'helsinki', null, '21240', 'finland', '90-224 8858', '90-224 8858');
insert into customers values ('wolza', 'wolski  zajazd', 'zbyszek piestrzeniewicz', 'owner', 'ul. filtrowa 68', 'warszawa', null, '01-012', 'poland', '(26) 642-7012', '(26) 642-7012');


--
-- data for name: employees; type: table data; schema: public; owner: postgres
--

insert into employees values (1, 'davolio', 'nancy', 'sales representative', 'ms.', '1948-12-08', '1992-05-01', '507 - 20th ave. e.\napt. 2a', 'seattle', 'wa', '98122', 'usa', '(206) 555-9857', '5467', '\x', 'education includes a ba in psychology from colorado state university in 1970.  she also completed "the art of the cold call."  nancy is a member of toastmasters international.', 2, 'http://accweb/emmployees/davolio.bmp');
insert into employees values (2, 'fuller', 'andrew', 'vice president, sales', 'dr.', '1952-02-19', '1992-08-14', '908 w. capital way', 'tacoma', 'wa', '98401', 'usa', '(206) 555-9482', '3457', '\x', 'andrew received his bts commercial in 1974 and a ph.d. in international marketing from the university of dallas in 1981.  he is fluent in french and italian and reads german.  he joined the company as a sales representative, was promoted to sales manager in january 1992 and to vice president of sales in march 1993.  andrew is a member of the sales management roundtable, the seattle chamber of commerce, and the pacific rim importers association.', null, 'http://accweb/emmployees/fuller.bmp');
insert into employees values (3, 'leverling', 'janet', 'sales representative', 'ms.', '1963-08-30', '1992-04-01', '722 moss bay blvd.', 'kirkland', 'wa', '98033', 'usa', '(206) 555-3412', '3355', '\x', 'janet has a bs degree in chemistry from boston college (1984).  she has also completed a certificate program in food retailing management.  janet was hired as a sales associate in 1991 and promoted to sales representative in february 1992.', 2, 'http://accweb/emmployees/leverling.bmp');
insert into employees values (4, 'peacock', 'margaret', 'sales representative', 'mrs.', '1937-09-19', '1993-05-03', '4110 old redmond rd.', 'redmond', 'wa', '98052', 'usa', '(206) 555-8122', '5176', '\x', 'margaret holds a ba in english literature from concordia college (1958) and an ma from the american institute of culinary arts (1966).  she was assigned to the london office temporarily from july through november 1992.', 2, 'http://accweb/emmployees/peacock.bmp');
insert into employees values (5, 'buchanan', 'steven', 'sales manager', 'mr.', '1955-03-04', '1993-10-17', '14 garrett hill', 'london', null, 'sw1 8jr', 'uk', '(71) 555-4848', '3453', '\x', 'steven buchanan graduated from st. andrews university, scotland, with a bsc degree in 1976.  upon joining the company as a sales representative in 1992, he spent 6 months in an orientation program at the seattle office and then returned to his permanent post in london.  he was promoted to sales manager in march 1993.  mr. buchanan has completed the courses "successful telemarketing" and "international sales management."  he is fluent in french.', 2, 'http://accweb/emmployees/buchanan.bmp');
insert into employees values (6, 'suyama', 'michael', 'sales representative', 'mr.', '1963-07-02', '1993-10-17', 'coventry house\nminer rd.', 'london', null, 'ec2 7jr', 'uk', '(71) 555-7773', '428', '\x', 'michael is a graduate of sussex university (ma, economics, 1983) and the university of california at los angeles (mba, marketing, 1986).  he has also taken the courses "multi-cultural selling" and "time management for the sales professional."  he is fluent in japanese and can read and write french, portuguese, and spanish.', 5, 'http://accweb/emmployees/davolio.bmp');
insert into employees values (7, 'king', 'robert', 'sales representative', 'mr.', '1960-05-29', '1994-01-02', 'edgeham hollow\nwinchester way', 'london', null, 'rg1 9sp', 'uk', '(71) 555-5598', '465', '\x', 'robert king served in the peace corps and traveled extensively before completing his degree in english at the university of michigan in 1992, the year he joined the company.  after completing a course entitled "selling in europe," he was transferred to the london office in march 1993.', 5, 'http://accweb/emmployees/davolio.bmp');
insert into employees values (8, 'callahan', 'laura', 'inside sales coordinator', 'ms.', '1958-01-09', '1994-03-05', '4726 - 11th ave. n.e.', 'seattle', 'wa', '98105', 'usa', '(206) 555-1189', '2344', '\x', 'laura received a ba in psychology from the university of washington.  she has also completed a course in business french.  she reads and writes french.', 2, 'http://accweb/emmployees/davolio.bmp');
insert into employees values (9, 'dodsworth', 'anne', 'sales representative', 'ms.', '1966-01-27', '1994-11-15', '7 houndstooth rd.', 'london', null, 'wg2 7lt', 'uk', '(71) 555-4444', '452', '\x', 'anne has a ba degree in english from st. lawrence college.  she is fluent in french and german.', 5, 'http://accweb/emmployees/davolio.bmp');


--
-- data for name: employeeterritories; type: table data; schema: public; owner: postgres
--

insert into employeeterritories values (1, '06897');
insert into employeeterritories values (1, '19713');
insert into employeeterritories values (2, '01581');
insert into employeeterritories values (2, '01730');
insert into employeeterritories values (2, '01833');
insert into employeeterritories values (2, '02116');
insert into employeeterritories values (2, '02139');
insert into employeeterritories values (2, '02184');
insert into employeeterritories values (2, '40222');
insert into employeeterritories values (3, '30346');
insert into employeeterritories values (3, '31406');
insert into employeeterritories values (3, '32859');
insert into employeeterritories values (3, '33607');
insert into employeeterritories values (4, '20852');
insert into employeeterritories values (4, '27403');
insert into employeeterritories values (4, '27511');
insert into employeeterritories values (5, '02903');
insert into employeeterritories values (5, '07960');
insert into employeeterritories values (5, '08837');
insert into employeeterritories values (5, '10019');
insert into employeeterritories values (5, '10038');
insert into employeeterritories values (5, '11747');
insert into employeeterritories values (5, '14450');
insert into employeeterritories values (6, '85014');
insert into employeeterritories values (6, '85251');
insert into employeeterritories values (6, '98004');
insert into employeeterritories values (6, '98052');
insert into employeeterritories values (6, '98104');
insert into employeeterritories values (7, '60179');
insert into employeeterritories values (7, '60601');
insert into employeeterritories values (7, '80202');
insert into employeeterritories values (7, '80909');
insert into employeeterritories values (7, '90405');
insert into employeeterritories values (7, '94025');
insert into employeeterritories values (7, '94105');
insert into employeeterritories values (7, '95008');
insert into employeeterritories values (7, '95054');
insert into employeeterritories values (7, '95060');
insert into employeeterritories values (8, '19428');
insert into employeeterritories values (8, '44122');
insert into employeeterritories values (8, '45839');
insert into employeeterritories values (8, '53404');
insert into employeeterritories values (9, '03049');
insert into employeeterritories values (9, '03801');
insert into employeeterritories values (9, '48075');
insert into employeeterritories values (9, '48084');
insert into employeeterritories values (9, '48304');
insert into employeeterritories values (9, '55113');
insert into employeeterritories values (9, '55439');


--
-- data for name: order_details; type: table data; schema: public; owner: postgres
--

insert into order_details values (10248, 11, 14, 12, 0);
insert into order_details values (10248, 42, 9.80000019, 10, 0);
insert into order_details values (10248, 72, 34.7999992, 5, 0);
insert into order_details values (10249, 14, 18.6000004, 9, 0);
insert into order_details values (10249, 51, 42.4000015, 40, 0);
insert into order_details values (10250, 41, 7.69999981, 10, 0);
insert into order_details values (10250, 51, 42.4000015, 35, 0.150000006);
insert into order_details values (10250, 65, 16.7999992, 15, 0.150000006);
insert into order_details values (10251, 22, 16.7999992, 6, 0.0500000007);
insert into order_details values (10251, 57, 15.6000004, 15, 0.0500000007);
insert into order_details values (10251, 65, 16.7999992, 20, 0);
insert into order_details values (10252, 20, 64.8000031, 40, 0.0500000007);
insert into order_details values (10252, 33, 2, 25, 0.0500000007);
insert into order_details values (10252, 60, 27.2000008, 40, 0);
insert into order_details values (10253, 31, 10, 20, 0);
insert into order_details values (10253, 39, 14.3999996, 42, 0);
insert into order_details values (10253, 49, 16, 40, 0);
insert into order_details values (10254, 24, 3.5999999, 15, 0.150000006);
insert into order_details values (10254, 55, 19.2000008, 21, 0.150000006);
insert into order_details values (10254, 74, 8, 21, 0);
insert into order_details values (10255, 2, 15.1999998, 20, 0);
insert into order_details values (10255, 16, 13.8999996, 35, 0);
insert into order_details values (10255, 36, 15.1999998, 25, 0);
insert into order_details values (10255, 59, 44, 30, 0);
insert into order_details values (10256, 53, 26.2000008, 15, 0);
insert into order_details values (10256, 77, 10.3999996, 12, 0);
insert into order_details values (10257, 27, 35.0999985, 25, 0);
insert into order_details values (10257, 39, 14.3999996, 6, 0);
insert into order_details values (10257, 77, 10.3999996, 15, 0);
insert into order_details values (10258, 2, 15.1999998, 50, 0.200000003);
insert into order_details values (10258, 5, 17, 65, 0.200000003);
insert into order_details values (10258, 32, 25.6000004, 6, 0.200000003);
insert into order_details values (10259, 21, 8, 10, 0);
insert into order_details values (10259, 37, 20.7999992, 1, 0);
insert into order_details values (10260, 41, 7.69999981, 16, 0.25);
insert into order_details values (10260, 57, 15.6000004, 50, 0);
insert into order_details values (10260, 62, 39.4000015, 15, 0.25);
insert into order_details values (10260, 70, 12, 21, 0.25);
insert into order_details values (10261, 21, 8, 20, 0);
insert into order_details values (10261, 35, 14.3999996, 20, 0);
insert into order_details values (10262, 5, 17, 12, 0.200000003);
insert into order_details values (10262, 7, 24, 15, 0);
insert into order_details values (10262, 56, 30.3999996, 2, 0);
insert into order_details values (10263, 16, 13.8999996, 60, 0.25);
insert into order_details values (10263, 24, 3.5999999, 28, 0);
insert into order_details values (10263, 30, 20.7000008, 60, 0.25);
insert into order_details values (10263, 74, 8, 36, 0.25);
insert into order_details values (10264, 2, 15.1999998, 35, 0);
insert into order_details values (10264, 41, 7.69999981, 25, 0.150000006);
insert into order_details values (10265, 17, 31.2000008, 30, 0);
insert into order_details values (10265, 70, 12, 20, 0);
insert into order_details values (10266, 12, 30.3999996, 12, 0.0500000007);
insert into order_details values (10267, 40, 14.6999998, 50, 0);
insert into order_details values (10267, 59, 44, 70, 0.150000006);
insert into order_details values (10267, 76, 14.3999996, 15, 0.150000006);
insert into order_details values (10268, 29, 99, 10, 0);
insert into order_details values (10268, 72, 27.7999992, 4, 0);
insert into order_details values (10269, 33, 2, 60, 0.0500000007);
insert into order_details values (10269, 72, 27.7999992, 20, 0.0500000007);
insert into order_details values (10270, 36, 15.1999998, 30, 0);
insert into order_details values (10270, 43, 36.7999992, 25, 0);
insert into order_details values (10271, 33, 2, 24, 0);
insert into order_details values (10272, 20, 64.8000031, 6, 0);
insert into order_details values (10272, 31, 10, 40, 0);
insert into order_details values (10272, 72, 27.7999992, 24, 0);
insert into order_details values (10273, 10, 24.7999992, 24, 0.0500000007);
insert into order_details values (10273, 31, 10, 15, 0.0500000007);
insert into order_details values (10273, 33, 2, 20, 0);
insert into order_details values (10273, 40, 14.6999998, 60, 0.0500000007);
insert into order_details values (10273, 76, 14.3999996, 33, 0.0500000007);
insert into order_details values (10274, 71, 17.2000008, 20, 0);
insert into order_details values (10274, 72, 27.7999992, 7, 0);
insert into order_details values (10275, 24, 3.5999999, 12, 0.0500000007);
insert into order_details values (10275, 59, 44, 6, 0.0500000007);
insert into order_details values (10276, 10, 24.7999992, 15, 0);
insert into order_details values (10276, 13, 4.80000019, 10, 0);
insert into order_details values (10277, 28, 36.4000015, 20, 0);
insert into order_details values (10277, 62, 39.4000015, 12, 0);
insert into order_details values (10278, 44, 15.5, 16, 0);
insert into order_details values (10278, 59, 44, 15, 0);
insert into order_details values (10278, 63, 35.0999985, 8, 0);
insert into order_details values (10278, 73, 12, 25, 0);
insert into order_details values (10279, 17, 31.2000008, 15, 0.25);
insert into order_details values (10280, 24, 3.5999999, 12, 0);
insert into order_details values (10280, 55, 19.2000008, 20, 0);
insert into order_details values (10280, 75, 6.19999981, 30, 0);
insert into order_details values (10281, 19, 7.30000019, 1, 0);
insert into order_details values (10281, 24, 3.5999999, 6, 0);
insert into order_details values (10281, 35, 14.3999996, 4, 0);
insert into order_details values (10282, 30, 20.7000008, 6, 0);
insert into order_details values (10282, 57, 15.6000004, 2, 0);
insert into order_details values (10283, 15, 12.3999996, 20, 0);
insert into order_details values (10283, 19, 7.30000019, 18, 0);
insert into order_details values (10283, 60, 27.2000008, 35, 0);
insert into order_details values (10283, 72, 27.7999992, 3, 0);
insert into order_details values (10284, 27, 35.0999985, 15, 0.25);
insert into order_details values (10284, 44, 15.5, 21, 0);
insert into order_details values (10284, 60, 27.2000008, 20, 0.25);
insert into order_details values (10284, 67, 11.1999998, 5, 0.25);
insert into order_details values (10285, 1, 14.3999996, 45, 0.200000003);
insert into order_details values (10285, 40, 14.6999998, 40, 0.200000003);
insert into order_details values (10285, 53, 26.2000008, 36, 0.200000003);
insert into order_details values (10286, 35, 14.3999996, 100, 0);
insert into order_details values (10286, 62, 39.4000015, 40, 0);
insert into order_details values (10287, 16, 13.8999996, 40, 0.150000006);
insert into order_details values (10287, 34, 11.1999998, 20, 0);
insert into order_details values (10287, 46, 9.60000038, 15, 0.150000006);
insert into order_details values (10288, 54, 5.9000001, 10, 0.100000001);
insert into order_details values (10288, 68, 10, 3, 0.100000001);
insert into order_details values (10289, 3, 8, 30, 0);
insert into order_details values (10289, 64, 26.6000004, 9, 0);
insert into order_details values (10290, 5, 17, 20, 0);
insert into order_details values (10290, 29, 99, 15, 0);
insert into order_details values (10290, 49, 16, 15, 0);
insert into order_details values (10290, 77, 10.3999996, 10, 0);
insert into order_details values (10291, 13, 4.80000019, 20, 0.100000001);
insert into order_details values (10291, 44, 15.5, 24, 0.100000001);
insert into order_details values (10291, 51, 42.4000015, 2, 0.100000001);
insert into order_details values (10292, 20, 64.8000031, 20, 0);
insert into order_details values (10293, 18, 50, 12, 0);
insert into order_details values (10293, 24, 3.5999999, 10, 0);
insert into order_details values (10293, 63, 35.0999985, 5, 0);
insert into order_details values (10293, 75, 6.19999981, 6, 0);
insert into order_details values (10294, 1, 14.3999996, 18, 0);
insert into order_details values (10294, 17, 31.2000008, 15, 0);
insert into order_details values (10294, 43, 36.7999992, 15, 0);
insert into order_details values (10294, 60, 27.2000008, 21, 0);
insert into order_details values (10294, 75, 6.19999981, 6, 0);
insert into order_details values (10295, 56, 30.3999996, 4, 0);
insert into order_details values (10296, 11, 16.7999992, 12, 0);
insert into order_details values (10296, 16, 13.8999996, 30, 0);
insert into order_details values (10296, 69, 28.7999992, 15, 0);
insert into order_details values (10297, 39, 14.3999996, 60, 0);
insert into order_details values (10297, 72, 27.7999992, 20, 0);
insert into order_details values (10298, 2, 15.1999998, 40, 0);
insert into order_details values (10298, 36, 15.1999998, 40, 0.25);
insert into order_details values (10298, 59, 44, 30, 0.25);
insert into order_details values (10298, 62, 39.4000015, 15, 0);
insert into order_details values (10299, 19, 7.30000019, 15, 0);
insert into order_details values (10299, 70, 12, 20, 0);
insert into order_details values (10300, 66, 13.6000004, 30, 0);
insert into order_details values (10300, 68, 10, 20, 0);
insert into order_details values (10301, 40, 14.6999998, 10, 0);
insert into order_details values (10301, 56, 30.3999996, 20, 0);
insert into order_details values (10302, 17, 31.2000008, 40, 0);
insert into order_details values (10302, 28, 36.4000015, 28, 0);
insert into order_details values (10302, 43, 36.7999992, 12, 0);
insert into order_details values (10303, 40, 14.6999998, 40, 0.100000001);
insert into order_details values (10303, 65, 16.7999992, 30, 0.100000001);
insert into order_details values (10303, 68, 10, 15, 0.100000001);
insert into order_details values (10304, 49, 16, 30, 0);
insert into order_details values (10304, 59, 44, 10, 0);
insert into order_details values (10304, 71, 17.2000008, 2, 0);
insert into order_details values (10305, 18, 50, 25, 0.100000001);
insert into order_details values (10305, 29, 99, 25, 0.100000001);
insert into order_details values (10305, 39, 14.3999996, 30, 0.100000001);
insert into order_details values (10306, 30, 20.7000008, 10, 0);
insert into order_details values (10306, 53, 26.2000008, 10, 0);
insert into order_details values (10306, 54, 5.9000001, 5, 0);
insert into order_details values (10307, 62, 39.4000015, 10, 0);
insert into order_details values (10307, 68, 10, 3, 0);
insert into order_details values (10308, 69, 28.7999992, 1, 0);
insert into order_details values (10308, 70, 12, 5, 0);
insert into order_details values (10309, 4, 17.6000004, 20, 0);
insert into order_details values (10309, 6, 20, 30, 0);
insert into order_details values (10309, 42, 11.1999998, 2, 0);
insert into order_details values (10309, 43, 36.7999992, 20, 0);
insert into order_details values (10309, 71, 17.2000008, 3, 0);
insert into order_details values (10310, 16, 13.8999996, 10, 0);
insert into order_details values (10310, 62, 39.4000015, 5, 0);
insert into order_details values (10311, 42, 11.1999998, 6, 0);
insert into order_details values (10311, 69, 28.7999992, 7, 0);
insert into order_details values (10312, 28, 36.4000015, 4, 0);
insert into order_details values (10312, 43, 36.7999992, 24, 0);
insert into order_details values (10312, 53, 26.2000008, 20, 0);
insert into order_details values (10312, 75, 6.19999981, 10, 0);
insert into order_details values (10313, 36, 15.1999998, 12, 0);
insert into order_details values (10314, 32, 25.6000004, 40, 0.100000001);
insert into order_details values (10314, 58, 10.6000004, 30, 0.100000001);
insert into order_details values (10314, 62, 39.4000015, 25, 0.100000001);
insert into order_details values (10315, 34, 11.1999998, 14, 0);
insert into order_details values (10315, 70, 12, 30, 0);
insert into order_details values (10316, 41, 7.69999981, 10, 0);
insert into order_details values (10316, 62, 39.4000015, 70, 0);
insert into order_details values (10317, 1, 14.3999996, 20, 0);
insert into order_details values (10318, 41, 7.69999981, 20, 0);
insert into order_details values (10318, 76, 14.3999996, 6, 0);
insert into order_details values (10319, 17, 31.2000008, 8, 0);
insert into order_details values (10319, 28, 36.4000015, 14, 0);
insert into order_details values (10319, 76, 14.3999996, 30, 0);
insert into order_details values (10320, 71, 17.2000008, 30, 0);
insert into order_details values (10321, 35, 14.3999996, 10, 0);
insert into order_details values (10322, 52, 5.5999999, 20, 0);
insert into order_details values (10323, 15, 12.3999996, 5, 0);
insert into order_details values (10323, 25, 11.1999998, 4, 0);
insert into order_details values (10323, 39, 14.3999996, 4, 0);
insert into order_details values (10324, 16, 13.8999996, 21, 0.150000006);
insert into order_details values (10324, 35, 14.3999996, 70, 0.150000006);
insert into order_details values (10324, 46, 9.60000038, 30, 0);
insert into order_details values (10324, 59, 44, 40, 0.150000006);
insert into order_details values (10324, 63, 35.0999985, 80, 0.150000006);
insert into order_details values (10325, 6, 20, 6, 0);
insert into order_details values (10325, 13, 4.80000019, 12, 0);
insert into order_details values (10325, 14, 18.6000004, 9, 0);
insert into order_details values (10325, 31, 10, 4, 0);
insert into order_details values (10325, 72, 27.7999992, 40, 0);
insert into order_details values (10326, 4, 17.6000004, 24, 0);
insert into order_details values (10326, 57, 15.6000004, 16, 0);
insert into order_details values (10326, 75, 6.19999981, 50, 0);
insert into order_details values (10327, 2, 15.1999998, 25, 0.200000003);
insert into order_details values (10327, 11, 16.7999992, 50, 0.200000003);
insert into order_details values (10327, 30, 20.7000008, 35, 0.200000003);
insert into order_details values (10327, 58, 10.6000004, 30, 0.200000003);
insert into order_details values (10328, 59, 44, 9, 0);
insert into order_details values (10328, 65, 16.7999992, 40, 0);
insert into order_details values (10328, 68, 10, 10, 0);
insert into order_details values (10329, 19, 7.30000019, 10, 0.0500000007);
insert into order_details values (10329, 30, 20.7000008, 8, 0.0500000007);
insert into order_details values (10329, 38, 210.800003, 20, 0.0500000007);
insert into order_details values (10329, 56, 30.3999996, 12, 0.0500000007);
insert into order_details values (10330, 26, 24.8999996, 50, 0.150000006);
insert into order_details values (10330, 72, 27.7999992, 25, 0.150000006);
insert into order_details values (10331, 54, 5.9000001, 15, 0);
insert into order_details values (10332, 18, 50, 40, 0.200000003);
insert into order_details values (10332, 42, 11.1999998, 10, 0.200000003);
insert into order_details values (10332, 47, 7.5999999, 16, 0.200000003);
insert into order_details values (10333, 14, 18.6000004, 10, 0);
insert into order_details values (10333, 21, 8, 10, 0.100000001);
insert into order_details values (10333, 71, 17.2000008, 40, 0.100000001);
insert into order_details values (10334, 52, 5.5999999, 8, 0);
insert into order_details values (10334, 68, 10, 10, 0);
insert into order_details values (10335, 2, 15.1999998, 7, 0.200000003);
insert into order_details values (10335, 31, 10, 25, 0.200000003);
insert into order_details values (10335, 32, 25.6000004, 6, 0.200000003);
insert into order_details values (10335, 51, 42.4000015, 48, 0.200000003);
insert into order_details values (10336, 4, 17.6000004, 18, 0.100000001);
insert into order_details values (10337, 23, 7.19999981, 40, 0);
insert into order_details values (10337, 26, 24.8999996, 24, 0);
insert into order_details values (10337, 36, 15.1999998, 20, 0);
insert into order_details values (10337, 37, 20.7999992, 28, 0);
insert into order_details values (10337, 72, 27.7999992, 25, 0);
insert into order_details values (10338, 17, 31.2000008, 20, 0);
insert into order_details values (10338, 30, 20.7000008, 15, 0);
insert into order_details values (10339, 4, 17.6000004, 10, 0);
insert into order_details values (10339, 17, 31.2000008, 70, 0.0500000007);
insert into order_details values (10339, 62, 39.4000015, 28, 0);
insert into order_details values (10340, 18, 50, 20, 0.0500000007);
insert into order_details values (10340, 41, 7.69999981, 12, 0.0500000007);
insert into order_details values (10340, 43, 36.7999992, 40, 0.0500000007);
insert into order_details values (10341, 33, 2, 8, 0);
insert into order_details values (10341, 59, 44, 9, 0.150000006);
insert into order_details values (10342, 2, 15.1999998, 24, 0.200000003);
insert into order_details values (10342, 31, 10, 56, 0.200000003);
insert into order_details values (10342, 36, 15.1999998, 40, 0.200000003);
insert into order_details values (10342, 55, 19.2000008, 40, 0.200000003);
insert into order_details values (10343, 64, 26.6000004, 50, 0);
insert into order_details values (10343, 68, 10, 4, 0.0500000007);
insert into order_details values (10343, 76, 14.3999996, 15, 0);
insert into order_details values (10344, 4, 17.6000004, 35, 0);
insert into order_details values (10344, 8, 32, 70, 0.25);
insert into order_details values (10345, 8, 32, 70, 0);
insert into order_details values (10345, 19, 7.30000019, 80, 0);
insert into order_details values (10345, 42, 11.1999998, 9, 0);
insert into order_details values (10346, 17, 31.2000008, 36, 0.100000001);
insert into order_details values (10346, 56, 30.3999996, 20, 0);
insert into order_details values (10347, 25, 11.1999998, 10, 0);
insert into order_details values (10347, 39, 14.3999996, 50, 0.150000006);
insert into order_details values (10347, 40, 14.6999998, 4, 0);
insert into order_details values (10347, 75, 6.19999981, 6, 0.150000006);
insert into order_details values (10348, 1, 14.3999996, 15, 0.150000006);
insert into order_details values (10348, 23, 7.19999981, 25, 0);
insert into order_details values (10349, 54, 5.9000001, 24, 0);
insert into order_details values (10350, 50, 13, 15, 0.100000001);
insert into order_details values (10350, 69, 28.7999992, 18, 0.100000001);
insert into order_details values (10351, 38, 210.800003, 20, 0.0500000007);
insert into order_details values (10351, 41, 7.69999981, 13, 0);
insert into order_details values (10351, 44, 15.5, 77, 0.0500000007);
insert into order_details values (10351, 65, 16.7999992, 10, 0.0500000007);
insert into order_details values (10352, 24, 3.5999999, 10, 0);
insert into order_details values (10352, 54, 5.9000001, 20, 0.150000006);
insert into order_details values (10353, 11, 16.7999992, 12, 0.200000003);
insert into order_details values (10353, 38, 210.800003, 50, 0.200000003);
insert into order_details values (10354, 1, 14.3999996, 12, 0);
insert into order_details values (10354, 29, 99, 4, 0);
insert into order_details values (10355, 24, 3.5999999, 25, 0);
insert into order_details values (10355, 57, 15.6000004, 25, 0);
insert into order_details values (10356, 31, 10, 30, 0);
insert into order_details values (10356, 55, 19.2000008, 12, 0);
insert into order_details values (10356, 69, 28.7999992, 20, 0);
insert into order_details values (10357, 10, 24.7999992, 30, 0.200000003);
insert into order_details values (10357, 26, 24.8999996, 16, 0);
insert into order_details values (10357, 60, 27.2000008, 8, 0.200000003);
insert into order_details values (10358, 24, 3.5999999, 10, 0.0500000007);
insert into order_details values (10358, 34, 11.1999998, 10, 0.0500000007);
insert into order_details values (10358, 36, 15.1999998, 20, 0.0500000007);
insert into order_details values (10359, 16, 13.8999996, 56, 0.0500000007);
insert into order_details values (10359, 31, 10, 70, 0.0500000007);
insert into order_details values (10359, 60, 27.2000008, 80, 0.0500000007);
insert into order_details values (10360, 28, 36.4000015, 30, 0);
insert into order_details values (10360, 29, 99, 35, 0);
insert into order_details values (10360, 38, 210.800003, 10, 0);
insert into order_details values (10360, 49, 16, 35, 0);
insert into order_details values (10360, 54, 5.9000001, 28, 0);
insert into order_details values (10361, 39, 14.3999996, 54, 0.100000001);
insert into order_details values (10361, 60, 27.2000008, 55, 0.100000001);
insert into order_details values (10362, 25, 11.1999998, 50, 0);
insert into order_details values (10362, 51, 42.4000015, 20, 0);
insert into order_details values (10362, 54, 5.9000001, 24, 0);
insert into order_details values (10363, 31, 10, 20, 0);
insert into order_details values (10363, 75, 6.19999981, 12, 0);
insert into order_details values (10363, 76, 14.3999996, 12, 0);
insert into order_details values (10364, 69, 28.7999992, 30, 0);
insert into order_details values (10364, 71, 17.2000008, 5, 0);
insert into order_details values (10365, 11, 16.7999992, 24, 0);
insert into order_details values (10366, 65, 16.7999992, 5, 0);
insert into order_details values (10366, 77, 10.3999996, 5, 0);
insert into order_details values (10367, 34, 11.1999998, 36, 0);
insert into order_details values (10367, 54, 5.9000001, 18, 0);
insert into order_details values (10367, 65, 16.7999992, 15, 0);
insert into order_details values (10367, 77, 10.3999996, 7, 0);
insert into order_details values (10368, 21, 8, 5, 0.100000001);
insert into order_details values (10368, 28, 36.4000015, 13, 0.100000001);
insert into order_details values (10368, 57, 15.6000004, 25, 0);
insert into order_details values (10368, 64, 26.6000004, 35, 0.100000001);
insert into order_details values (10369, 29, 99, 20, 0);
insert into order_details values (10369, 56, 30.3999996, 18, 0.25);
insert into order_details values (10370, 1, 14.3999996, 15, 0.150000006);
insert into order_details values (10370, 64, 26.6000004, 30, 0);
insert into order_details values (10370, 74, 8, 20, 0.150000006);
insert into order_details values (10371, 36, 15.1999998, 6, 0.200000003);
insert into order_details values (10372, 20, 64.8000031, 12, 0.25);
insert into order_details values (10372, 38, 210.800003, 40, 0.25);
insert into order_details values (10372, 60, 27.2000008, 70, 0.25);
insert into order_details values (10372, 72, 27.7999992, 42, 0.25);
insert into order_details values (10373, 58, 10.6000004, 80, 0.200000003);
insert into order_details values (10373, 71, 17.2000008, 50, 0.200000003);
insert into order_details values (10374, 31, 10, 30, 0);
insert into order_details values (10374, 58, 10.6000004, 15, 0);
insert into order_details values (10375, 14, 18.6000004, 15, 0);
insert into order_details values (10375, 54, 5.9000001, 10, 0);
insert into order_details values (10376, 31, 10, 42, 0.0500000007);
insert into order_details values (10377, 28, 36.4000015, 20, 0.150000006);
insert into order_details values (10377, 39, 14.3999996, 20, 0.150000006);
insert into order_details values (10378, 71, 17.2000008, 6, 0);
insert into order_details values (10379, 41, 7.69999981, 8, 0.100000001);
insert into order_details values (10379, 63, 35.0999985, 16, 0.100000001);
insert into order_details values (10379, 65, 16.7999992, 20, 0.100000001);
insert into order_details values (10380, 30, 20.7000008, 18, 0.100000001);
insert into order_details values (10380, 53, 26.2000008, 20, 0.100000001);
insert into order_details values (10380, 60, 27.2000008, 6, 0.100000001);
insert into order_details values (10380, 70, 12, 30, 0);
insert into order_details values (10381, 74, 8, 14, 0);
insert into order_details values (10382, 5, 17, 32, 0);
insert into order_details values (10382, 18, 50, 9, 0);
insert into order_details values (10382, 29, 99, 14, 0);
insert into order_details values (10382, 33, 2, 60, 0);
insert into order_details values (10382, 74, 8, 50, 0);
insert into order_details values (10383, 13, 4.80000019, 20, 0);
insert into order_details values (10383, 50, 13, 15, 0);
insert into order_details values (10383, 56, 30.3999996, 20, 0);
insert into order_details values (10384, 20, 64.8000031, 28, 0);
insert into order_details values (10384, 60, 27.2000008, 15, 0);
insert into order_details values (10385, 7, 24, 10, 0.200000003);
insert into order_details values (10385, 60, 27.2000008, 20, 0.200000003);
insert into order_details values (10385, 68, 10, 8, 0.200000003);
insert into order_details values (10386, 24, 3.5999999, 15, 0);
insert into order_details values (10386, 34, 11.1999998, 10, 0);
insert into order_details values (10387, 24, 3.5999999, 15, 0);
insert into order_details values (10387, 28, 36.4000015, 6, 0);
insert into order_details values (10387, 59, 44, 12, 0);
insert into order_details values (10387, 71, 17.2000008, 15, 0);
insert into order_details values (10388, 45, 7.5999999, 15, 0.200000003);
insert into order_details values (10388, 52, 5.5999999, 20, 0.200000003);
insert into order_details values (10388, 53, 26.2000008, 40, 0);
insert into order_details values (10389, 10, 24.7999992, 16, 0);
insert into order_details values (10389, 55, 19.2000008, 15, 0);
insert into order_details values (10389, 62, 39.4000015, 20, 0);
insert into order_details values (10389, 70, 12, 30, 0);
insert into order_details values (10390, 31, 10, 60, 0.100000001);
insert into order_details values (10390, 35, 14.3999996, 40, 0.100000001);
insert into order_details values (10390, 46, 9.60000038, 45, 0);
insert into order_details values (10390, 72, 27.7999992, 24, 0.100000001);
insert into order_details values (10391, 13, 4.80000019, 18, 0);
insert into order_details values (10392, 69, 28.7999992, 50, 0);
insert into order_details values (10393, 2, 15.1999998, 25, 0.25);
insert into order_details values (10393, 14, 18.6000004, 42, 0.25);
insert into order_details values (10393, 25, 11.1999998, 7, 0.25);
insert into order_details values (10393, 26, 24.8999996, 70, 0.25);
insert into order_details values (10393, 31, 10, 32, 0);
insert into order_details values (10394, 13, 4.80000019, 10, 0);
insert into order_details values (10394, 62, 39.4000015, 10, 0);
insert into order_details values (10395, 46, 9.60000038, 28, 0.100000001);
insert into order_details values (10395, 53, 26.2000008, 70, 0.100000001);
insert into order_details values (10395, 69, 28.7999992, 8, 0);
insert into order_details values (10396, 23, 7.19999981, 40, 0);
insert into order_details values (10396, 71, 17.2000008, 60, 0);
insert into order_details values (10396, 72, 27.7999992, 21, 0);
insert into order_details values (10397, 21, 8, 10, 0.150000006);
insert into order_details values (10397, 51, 42.4000015, 18, 0.150000006);
insert into order_details values (10398, 35, 14.3999996, 30, 0);
insert into order_details values (10398, 55, 19.2000008, 120, 0.100000001);
insert into order_details values (10399, 68, 10, 60, 0);
insert into order_details values (10399, 71, 17.2000008, 30, 0);
insert into order_details values (10399, 76, 14.3999996, 35, 0);
insert into order_details values (10399, 77, 10.3999996, 14, 0);
insert into order_details values (10400, 29, 99, 21, 0);
insert into order_details values (10400, 35, 14.3999996, 35, 0);
insert into order_details values (10400, 49, 16, 30, 0);
insert into order_details values (10401, 30, 20.7000008, 18, 0);
insert into order_details values (10401, 56, 30.3999996, 70, 0);
insert into order_details values (10401, 65, 16.7999992, 20, 0);
insert into order_details values (10401, 71, 17.2000008, 60, 0);
insert into order_details values (10402, 23, 7.19999981, 60, 0);
insert into order_details values (10402, 63, 35.0999985, 65, 0);
insert into order_details values (10403, 16, 13.8999996, 21, 0.150000006);
insert into order_details values (10403, 48, 10.1999998, 70, 0.150000006);
insert into order_details values (10404, 26, 24.8999996, 30, 0.0500000007);
insert into order_details values (10404, 42, 11.1999998, 40, 0.0500000007);
insert into order_details values (10404, 49, 16, 30, 0.0500000007);
insert into order_details values (10405, 3, 8, 50, 0);
insert into order_details values (10406, 1, 14.3999996, 10, 0);
insert into order_details values (10406, 21, 8, 30, 0.100000001);
insert into order_details values (10406, 28, 36.4000015, 42, 0.100000001);
insert into order_details values (10406, 36, 15.1999998, 5, 0.100000001);
insert into order_details values (10406, 40, 14.6999998, 2, 0.100000001);
insert into order_details values (10407, 11, 16.7999992, 30, 0);
insert into order_details values (10407, 69, 28.7999992, 15, 0);
insert into order_details values (10407, 71, 17.2000008, 15, 0);
insert into order_details values (10408, 37, 20.7999992, 10, 0);
insert into order_details values (10408, 54, 5.9000001, 6, 0);
insert into order_details values (10408, 62, 39.4000015, 35, 0);
insert into order_details values (10409, 14, 18.6000004, 12, 0);
insert into order_details values (10409, 21, 8, 12, 0);
insert into order_details values (10410, 33, 2, 49, 0);
insert into order_details values (10410, 59, 44, 16, 0);
insert into order_details values (10411, 41, 7.69999981, 25, 0.200000003);
insert into order_details values (10411, 44, 15.5, 40, 0.200000003);
insert into order_details values (10411, 59, 44, 9, 0.200000003);
insert into order_details values (10412, 14, 18.6000004, 20, 0.100000001);
insert into order_details values (10413, 1, 14.3999996, 24, 0);
insert into order_details values (10413, 62, 39.4000015, 40, 0);
insert into order_details values (10413, 76, 14.3999996, 14, 0);
insert into order_details values (10414, 19, 7.30000019, 18, 0.0500000007);
insert into order_details values (10414, 33, 2, 50, 0);
insert into order_details values (10415, 17, 31.2000008, 2, 0);
insert into order_details values (10415, 33, 2, 20, 0);
insert into order_details values (10416, 19, 7.30000019, 20, 0);
insert into order_details values (10416, 53, 26.2000008, 10, 0);
insert into order_details values (10416, 57, 15.6000004, 20, 0);
insert into order_details values (10417, 38, 210.800003, 50, 0);
insert into order_details values (10417, 46, 9.60000038, 2, 0.25);
insert into order_details values (10417, 68, 10, 36, 0.25);
insert into order_details values (10417, 77, 10.3999996, 35, 0);
insert into order_details values (10418, 2, 15.1999998, 60, 0);
insert into order_details values (10418, 47, 7.5999999, 55, 0);
insert into order_details values (10418, 61, 22.7999992, 16, 0);
insert into order_details values (10418, 74, 8, 15, 0);
insert into order_details values (10419, 60, 27.2000008, 60, 0.0500000007);
insert into order_details values (10419, 69, 28.7999992, 20, 0.0500000007);
insert into order_details values (10420, 9, 77.5999985, 20, 0.100000001);
insert into order_details values (10420, 13, 4.80000019, 2, 0.100000001);
insert into order_details values (10420, 70, 12, 8, 0.100000001);
insert into order_details values (10420, 73, 12, 20, 0.100000001);
insert into order_details values (10421, 19, 7.30000019, 4, 0.150000006);
insert into order_details values (10421, 26, 24.8999996, 30, 0);
insert into order_details values (10421, 53, 26.2000008, 15, 0.150000006);
insert into order_details values (10421, 77, 10.3999996, 10, 0.150000006);
insert into order_details values (10422, 26, 24.8999996, 2, 0);
insert into order_details values (10423, 31, 10, 14, 0);
insert into order_details values (10423, 59, 44, 20, 0);
insert into order_details values (10424, 35, 14.3999996, 60, 0.200000003);
insert into order_details values (10424, 38, 210.800003, 49, 0.200000003);
insert into order_details values (10424, 68, 10, 30, 0.200000003);
insert into order_details values (10425, 55, 19.2000008, 10, 0.25);
insert into order_details values (10425, 76, 14.3999996, 20, 0.25);
insert into order_details values (10426, 56, 30.3999996, 5, 0);
insert into order_details values (10426, 64, 26.6000004, 7, 0);
insert into order_details values (10427, 14, 18.6000004, 35, 0);
insert into order_details values (10428, 46, 9.60000038, 20, 0);
insert into order_details values (10429, 50, 13, 40, 0);
insert into order_details values (10429, 63, 35.0999985, 35, 0.25);
insert into order_details values (10430, 17, 31.2000008, 45, 0.200000003);
insert into order_details values (10430, 21, 8, 50, 0);
insert into order_details values (10430, 56, 30.3999996, 30, 0);
insert into order_details values (10430, 59, 44, 70, 0.200000003);
insert into order_details values (10431, 17, 31.2000008, 50, 0.25);
insert into order_details values (10431, 40, 14.6999998, 50, 0.25);
insert into order_details values (10431, 47, 7.5999999, 30, 0.25);
insert into order_details values (10432, 26, 24.8999996, 10, 0);
insert into order_details values (10432, 54, 5.9000001, 40, 0);
insert into order_details values (10433, 56, 30.3999996, 28, 0);
insert into order_details values (10434, 11, 16.7999992, 6, 0);
insert into order_details values (10434, 76, 14.3999996, 18, 0.150000006);
insert into order_details values (10435, 2, 15.1999998, 10, 0);
insert into order_details values (10435, 22, 16.7999992, 12, 0);
insert into order_details values (10435, 72, 27.7999992, 10, 0);
insert into order_details values (10436, 46, 9.60000038, 5, 0);
insert into order_details values (10436, 56, 30.3999996, 40, 0.100000001);
insert into order_details values (10436, 64, 26.6000004, 30, 0.100000001);
insert into order_details values (10436, 75, 6.19999981, 24, 0.100000001);
insert into order_details values (10437, 53, 26.2000008, 15, 0);
insert into order_details values (10438, 19, 7.30000019, 15, 0.200000003);
insert into order_details values (10438, 34, 11.1999998, 20, 0.200000003);
insert into order_details values (10438, 57, 15.6000004, 15, 0.200000003);
insert into order_details values (10439, 12, 30.3999996, 15, 0);
insert into order_details values (10439, 16, 13.8999996, 16, 0);
insert into order_details values (10439, 64, 26.6000004, 6, 0);
insert into order_details values (10439, 74, 8, 30, 0);
insert into order_details values (10440, 2, 15.1999998, 45, 0.150000006);
insert into order_details values (10440, 16, 13.8999996, 49, 0.150000006);
insert into order_details values (10440, 29, 99, 24, 0.150000006);
insert into order_details values (10440, 61, 22.7999992, 90, 0.150000006);
insert into order_details values (10441, 27, 35.0999985, 50, 0);
insert into order_details values (10442, 11, 16.7999992, 30, 0);
insert into order_details values (10442, 54, 5.9000001, 80, 0);
insert into order_details values (10442, 66, 13.6000004, 60, 0);
insert into order_details values (10443, 11, 16.7999992, 6, 0.200000003);
insert into order_details values (10443, 28, 36.4000015, 12, 0);
insert into order_details values (10444, 17, 31.2000008, 10, 0);
insert into order_details values (10444, 26, 24.8999996, 15, 0);
insert into order_details values (10444, 35, 14.3999996, 8, 0);
insert into order_details values (10444, 41, 7.69999981, 30, 0);
insert into order_details values (10445, 39, 14.3999996, 6, 0);
insert into order_details values (10445, 54, 5.9000001, 15, 0);
insert into order_details values (10446, 19, 7.30000019, 12, 0.100000001);
insert into order_details values (10446, 24, 3.5999999, 20, 0.100000001);
insert into order_details values (10446, 31, 10, 3, 0.100000001);
insert into order_details values (10446, 52, 5.5999999, 15, 0.100000001);
insert into order_details values (10447, 19, 7.30000019, 40, 0);
insert into order_details values (10447, 65, 16.7999992, 35, 0);
insert into order_details values (10447, 71, 17.2000008, 2, 0);
insert into order_details values (10448, 26, 24.8999996, 6, 0);
insert into order_details values (10448, 40, 14.6999998, 20, 0);
insert into order_details values (10449, 10, 24.7999992, 14, 0);
insert into order_details values (10449, 52, 5.5999999, 20, 0);
insert into order_details values (10449, 62, 39.4000015, 35, 0);
insert into order_details values (10450, 10, 24.7999992, 20, 0.200000003);
insert into order_details values (10450, 54, 5.9000001, 6, 0.200000003);
insert into order_details values (10451, 55, 19.2000008, 120, 0.100000001);
insert into order_details values (10451, 64, 26.6000004, 35, 0.100000001);
insert into order_details values (10451, 65, 16.7999992, 28, 0.100000001);
insert into order_details values (10451, 77, 10.3999996, 55, 0.100000001);
insert into order_details values (10452, 28, 36.4000015, 15, 0);
insert into order_details values (10452, 44, 15.5, 100, 0.0500000007);
insert into order_details values (10453, 48, 10.1999998, 15, 0.100000001);
insert into order_details values (10453, 70, 12, 25, 0.100000001);
insert into order_details values (10454, 16, 13.8999996, 20, 0.200000003);
insert into order_details values (10454, 33, 2, 20, 0.200000003);
insert into order_details values (10454, 46, 9.60000038, 10, 0.200000003);
insert into order_details values (10455, 39, 14.3999996, 20, 0);
insert into order_details values (10455, 53, 26.2000008, 50, 0);
insert into order_details values (10455, 61, 22.7999992, 25, 0);
insert into order_details values (10455, 71, 17.2000008, 30, 0);
insert into order_details values (10456, 21, 8, 40, 0.150000006);
insert into order_details values (10456, 49, 16, 21, 0.150000006);
insert into order_details values (10457, 59, 44, 36, 0);
insert into order_details values (10458, 26, 24.8999996, 30, 0);
insert into order_details values (10458, 28, 36.4000015, 30, 0);
insert into order_details values (10458, 43, 36.7999992, 20, 0);
insert into order_details values (10458, 56, 30.3999996, 15, 0);
insert into order_details values (10458, 71, 17.2000008, 50, 0);
insert into order_details values (10459, 7, 24, 16, 0.0500000007);
insert into order_details values (10459, 46, 9.60000038, 20, 0.0500000007);
insert into order_details values (10459, 72, 27.7999992, 40, 0);
insert into order_details values (10460, 68, 10, 21, 0.25);
insert into order_details values (10460, 75, 6.19999981, 4, 0.25);
insert into order_details values (10461, 21, 8, 40, 0.25);
insert into order_details values (10461, 30, 20.7000008, 28, 0.25);
insert into order_details values (10461, 55, 19.2000008, 60, 0.25);
insert into order_details values (10462, 13, 4.80000019, 1, 0);
insert into order_details values (10462, 23, 7.19999981, 21, 0);
insert into order_details values (10463, 19, 7.30000019, 21, 0);
insert into order_details values (10463, 42, 11.1999998, 50, 0);
insert into order_details values (10464, 4, 17.6000004, 16, 0.200000003);
insert into order_details values (10464, 43, 36.7999992, 3, 0);
insert into order_details values (10464, 56, 30.3999996, 30, 0.200000003);
insert into order_details values (10464, 60, 27.2000008, 20, 0);
insert into order_details values (10465, 24, 3.5999999, 25, 0);
insert into order_details values (10465, 29, 99, 18, 0.100000001);
insert into order_details values (10465, 40, 14.6999998, 20, 0);
insert into order_details values (10465, 45, 7.5999999, 30, 0.100000001);
insert into order_details values (10465, 50, 13, 25, 0);
insert into order_details values (10466, 11, 16.7999992, 10, 0);
insert into order_details values (10466, 46, 9.60000038, 5, 0);
insert into order_details values (10467, 24, 3.5999999, 28, 0);
insert into order_details values (10467, 25, 11.1999998, 12, 0);
insert into order_details values (10468, 30, 20.7000008, 8, 0);
insert into order_details values (10468, 43, 36.7999992, 15, 0);
insert into order_details values (10469, 2, 15.1999998, 40, 0.150000006);
insert into order_details values (10469, 16, 13.8999996, 35, 0.150000006);
insert into order_details values (10469, 44, 15.5, 2, 0.150000006);
insert into order_details values (10470, 18, 50, 30, 0);
insert into order_details values (10470, 23, 7.19999981, 15, 0);
insert into order_details values (10470, 64, 26.6000004, 8, 0);
insert into order_details values (10471, 7, 24, 30, 0);
insert into order_details values (10471, 56, 30.3999996, 20, 0);
insert into order_details values (10472, 24, 3.5999999, 80, 0.0500000007);
insert into order_details values (10472, 51, 42.4000015, 18, 0);
insert into order_details values (10473, 33, 2, 12, 0);
insert into order_details values (10473, 71, 17.2000008, 12, 0);
insert into order_details values (10474, 14, 18.6000004, 12, 0);
insert into order_details values (10474, 28, 36.4000015, 18, 0);
insert into order_details values (10474, 40, 14.6999998, 21, 0);
insert into order_details values (10474, 75, 6.19999981, 10, 0);
insert into order_details values (10475, 31, 10, 35, 0.150000006);
insert into order_details values (10475, 66, 13.6000004, 60, 0.150000006);
insert into order_details values (10475, 76, 14.3999996, 42, 0.150000006);
insert into order_details values (10476, 55, 19.2000008, 2, 0.0500000007);
insert into order_details values (10476, 70, 12, 12, 0);
insert into order_details values (10477, 1, 14.3999996, 15, 0);
insert into order_details values (10477, 21, 8, 21, 0.25);
insert into order_details values (10477, 39, 14.3999996, 20, 0.25);
insert into order_details values (10478, 10, 24.7999992, 20, 0.0500000007);
insert into order_details values (10479, 38, 210.800003, 30, 0);
insert into order_details values (10479, 53, 26.2000008, 28, 0);
insert into order_details values (10479, 59, 44, 60, 0);
insert into order_details values (10479, 64, 26.6000004, 30, 0);
insert into order_details values (10480, 47, 7.5999999, 30, 0);
insert into order_details values (10480, 59, 44, 12, 0);
insert into order_details values (10481, 49, 16, 24, 0);
insert into order_details values (10481, 60, 27.2000008, 40, 0);
insert into order_details values (10482, 40, 14.6999998, 10, 0);
insert into order_details values (10483, 34, 11.1999998, 35, 0.0500000007);
insert into order_details values (10483, 77, 10.3999996, 30, 0.0500000007);
insert into order_details values (10484, 21, 8, 14, 0);
insert into order_details values (10484, 40, 14.6999998, 10, 0);
insert into order_details values (10484, 51, 42.4000015, 3, 0);
insert into order_details values (10485, 2, 15.1999998, 20, 0.100000001);
insert into order_details values (10485, 3, 8, 20, 0.100000001);
insert into order_details values (10485, 55, 19.2000008, 30, 0.100000001);
insert into order_details values (10485, 70, 12, 60, 0.100000001);
insert into order_details values (10486, 11, 16.7999992, 5, 0);
insert into order_details values (10486, 51, 42.4000015, 25, 0);
insert into order_details values (10486, 74, 8, 16, 0);
insert into order_details values (10487, 19, 7.30000019, 5, 0);
insert into order_details values (10487, 26, 24.8999996, 30, 0);
insert into order_details values (10487, 54, 5.9000001, 24, 0.25);
insert into order_details values (10488, 59, 44, 30, 0);
insert into order_details values (10488, 73, 12, 20, 0.200000003);
insert into order_details values (10489, 11, 16.7999992, 15, 0.25);
insert into order_details values (10489, 16, 13.8999996, 18, 0);
insert into order_details values (10490, 59, 44, 60, 0);
insert into order_details values (10490, 68, 10, 30, 0);
insert into order_details values (10490, 75, 6.19999981, 36, 0);
insert into order_details values (10491, 44, 15.5, 15, 0.150000006);
insert into order_details values (10491, 77, 10.3999996, 7, 0.150000006);
insert into order_details values (10492, 25, 11.1999998, 60, 0.0500000007);
insert into order_details values (10492, 42, 11.1999998, 20, 0.0500000007);
insert into order_details values (10493, 65, 16.7999992, 15, 0.100000001);
insert into order_details values (10493, 66, 13.6000004, 10, 0.100000001);
insert into order_details values (10493, 69, 28.7999992, 10, 0.100000001);
insert into order_details values (10494, 56, 30.3999996, 30, 0);
insert into order_details values (10495, 23, 7.19999981, 10, 0);
insert into order_details values (10495, 41, 7.69999981, 20, 0);
insert into order_details values (10495, 77, 10.3999996, 5, 0);
insert into order_details values (10496, 31, 10, 20, 0.0500000007);
insert into order_details values (10497, 56, 30.3999996, 14, 0);
insert into order_details values (10497, 72, 27.7999992, 25, 0);
insert into order_details values (10497, 77, 10.3999996, 25, 0);
insert into order_details values (10498, 24, 4.5, 14, 0);
insert into order_details values (10498, 40, 18.3999996, 5, 0);
insert into order_details values (10498, 42, 14, 30, 0);
insert into order_details values (10499, 28, 45.5999985, 20, 0);
insert into order_details values (10499, 49, 20, 25, 0);
insert into order_details values (10500, 15, 15.5, 12, 0.0500000007);
insert into order_details values (10500, 28, 45.5999985, 8, 0.0500000007);
insert into order_details values (10501, 54, 7.44999981, 20, 0);
insert into order_details values (10502, 45, 9.5, 21, 0);
insert into order_details values (10502, 53, 32.7999992, 6, 0);
insert into order_details values (10502, 67, 14, 30, 0);
insert into order_details values (10503, 14, 23.25, 70, 0);
insert into order_details values (10503, 65, 21.0499992, 20, 0);
insert into order_details values (10504, 2, 19, 12, 0);
insert into order_details values (10504, 21, 10, 12, 0);
insert into order_details values (10504, 53, 32.7999992, 10, 0);
insert into order_details values (10504, 61, 28.5, 25, 0);
insert into order_details values (10505, 62, 49.2999992, 3, 0);
insert into order_details values (10506, 25, 14, 18, 0.100000001);
insert into order_details values (10506, 70, 15, 14, 0.100000001);
insert into order_details values (10507, 43, 46, 15, 0.150000006);
insert into order_details values (10507, 48, 12.75, 15, 0.150000006);
insert into order_details values (10508, 13, 6, 10, 0);
insert into order_details values (10508, 39, 18, 10, 0);
insert into order_details values (10509, 28, 45.5999985, 3, 0);
insert into order_details values (10510, 29, 123.790001, 36, 0);
insert into order_details values (10510, 75, 7.75, 36, 0.100000001);
insert into order_details values (10511, 4, 22, 50, 0.150000006);
insert into order_details values (10511, 7, 30, 50, 0.150000006);
insert into order_details values (10511, 8, 40, 10, 0.150000006);
insert into order_details values (10512, 24, 4.5, 10, 0.150000006);
insert into order_details values (10512, 46, 12, 9, 0.150000006);
insert into order_details values (10512, 47, 9.5, 6, 0.150000006);
insert into order_details values (10512, 60, 34, 12, 0.150000006);
insert into order_details values (10513, 21, 10, 40, 0.200000003);
insert into order_details values (10513, 32, 32, 50, 0.200000003);
insert into order_details values (10513, 61, 28.5, 15, 0.200000003);
insert into order_details values (10514, 20, 81, 39, 0);
insert into order_details values (10514, 28, 45.5999985, 35, 0);
insert into order_details values (10514, 56, 38, 70, 0);
insert into order_details values (10514, 65, 21.0499992, 39, 0);
insert into order_details values (10514, 75, 7.75, 50, 0);
insert into order_details values (10515, 9, 97, 16, 0.150000006);
insert into order_details values (10515, 16, 17.4500008, 50, 0);
insert into order_details values (10515, 27, 43.9000015, 120, 0);
insert into order_details values (10515, 33, 2.5, 16, 0.150000006);
insert into order_details values (10515, 60, 34, 84, 0.150000006);
insert into order_details values (10516, 18, 62.5, 25, 0.100000001);
insert into order_details values (10516, 41, 9.64999962, 80, 0.100000001);
insert into order_details values (10516, 42, 14, 20, 0);
insert into order_details values (10517, 52, 7, 6, 0);
insert into order_details values (10517, 59, 55, 4, 0);
insert into order_details values (10517, 70, 15, 6, 0);
insert into order_details values (10518, 24, 4.5, 5, 0);
insert into order_details values (10518, 38, 263.5, 15, 0);
insert into order_details values (10518, 44, 19.4500008, 9, 0);
insert into order_details values (10519, 10, 31, 16, 0.0500000007);
insert into order_details values (10519, 56, 38, 40, 0);
insert into order_details values (10519, 60, 34, 10, 0.0500000007);
insert into order_details values (10520, 24, 4.5, 8, 0);
insert into order_details values (10520, 53, 32.7999992, 5, 0);
insert into order_details values (10521, 35, 18, 3, 0);
insert into order_details values (10521, 41, 9.64999962, 10, 0);
insert into order_details values (10521, 68, 12.5, 6, 0);
insert into order_details values (10522, 1, 18, 40, 0.200000003);
insert into order_details values (10522, 8, 40, 24, 0);
insert into order_details values (10522, 30, 25.8899994, 20, 0.200000003);
insert into order_details values (10522, 40, 18.3999996, 25, 0.200000003);
insert into order_details values (10523, 17, 39, 25, 0.100000001);
insert into order_details values (10523, 20, 81, 15, 0.100000001);
insert into order_details values (10523, 37, 26, 18, 0.100000001);
insert into order_details values (10523, 41, 9.64999962, 6, 0.100000001);
insert into order_details values (10524, 10, 31, 2, 0);
insert into order_details values (10524, 30, 25.8899994, 10, 0);
insert into order_details values (10524, 43, 46, 60, 0);
insert into order_details values (10524, 54, 7.44999981, 15, 0);
insert into order_details values (10525, 36, 19, 30, 0);
insert into order_details values (10525, 40, 18.3999996, 15, 0.100000001);
insert into order_details values (10526, 1, 18, 8, 0.150000006);
insert into order_details values (10526, 13, 6, 10, 0);
insert into order_details values (10526, 56, 38, 30, 0.150000006);
insert into order_details values (10527, 4, 22, 50, 0.100000001);
insert into order_details values (10527, 36, 19, 30, 0.100000001);
insert into order_details values (10528, 11, 21, 3, 0);
insert into order_details values (10528, 33, 2.5, 8, 0.200000003);
insert into order_details values (10528, 72, 34.7999992, 9, 0);
insert into order_details values (10529, 55, 24, 14, 0);
insert into order_details values (10529, 68, 12.5, 20, 0);
insert into order_details values (10529, 69, 36, 10, 0);
insert into order_details values (10530, 17, 39, 40, 0);
insert into order_details values (10530, 43, 46, 25, 0);
insert into order_details values (10530, 61, 28.5, 20, 0);
insert into order_details values (10530, 76, 18, 50, 0);
insert into order_details values (10531, 59, 55, 2, 0);
insert into order_details values (10532, 30, 25.8899994, 15, 0);
insert into order_details values (10532, 66, 17, 24, 0);
insert into order_details values (10533, 4, 22, 50, 0.0500000007);
insert into order_details values (10533, 72, 34.7999992, 24, 0);
insert into order_details values (10533, 73, 15, 24, 0.0500000007);
insert into order_details values (10534, 30, 25.8899994, 10, 0);
insert into order_details values (10534, 40, 18.3999996, 10, 0.200000003);
insert into order_details values (10534, 54, 7.44999981, 10, 0.200000003);
insert into order_details values (10535, 11, 21, 50, 0.100000001);
insert into order_details values (10535, 40, 18.3999996, 10, 0.100000001);
insert into order_details values (10535, 57, 19.5, 5, 0.100000001);
insert into order_details values (10535, 59, 55, 15, 0.100000001);
insert into order_details values (10536, 12, 38, 15, 0.25);
insert into order_details values (10536, 31, 12.5, 20, 0);
insert into order_details values (10536, 33, 2.5, 30, 0);
insert into order_details values (10536, 60, 34, 35, 0.25);
insert into order_details values (10537, 31, 12.5, 30, 0);
insert into order_details values (10537, 51, 53, 6, 0);
insert into order_details values (10537, 58, 13.25, 20, 0);
insert into order_details values (10537, 72, 34.7999992, 21, 0);
insert into order_details values (10537, 73, 15, 9, 0);
insert into order_details values (10538, 70, 15, 7, 0);
insert into order_details values (10538, 72, 34.7999992, 1, 0);
insert into order_details values (10539, 13, 6, 8, 0);
insert into order_details values (10539, 21, 10, 15, 0);
insert into order_details values (10539, 33, 2.5, 15, 0);
insert into order_details values (10539, 49, 20, 6, 0);
insert into order_details values (10540, 3, 10, 60, 0);
insert into order_details values (10540, 26, 31.2299995, 40, 0);
insert into order_details values (10540, 38, 263.5, 30, 0);
insert into order_details values (10540, 68, 12.5, 35, 0);
insert into order_details values (10541, 24, 4.5, 35, 0.100000001);
insert into order_details values (10541, 38, 263.5, 4, 0.100000001);
insert into order_details values (10541, 65, 21.0499992, 36, 0.100000001);
insert into order_details values (10541, 71, 21.5, 9, 0.100000001);
insert into order_details values (10542, 11, 21, 15, 0.0500000007);
insert into order_details values (10542, 54, 7.44999981, 24, 0.0500000007);
insert into order_details values (10543, 12, 38, 30, 0.150000006);
insert into order_details values (10543, 23, 9, 70, 0.150000006);
insert into order_details values (10544, 28, 45.5999985, 7, 0);
insert into order_details values (10544, 67, 14, 7, 0);
insert into order_details values (10545, 11, 21, 10, 0);
insert into order_details values (10546, 7, 30, 10, 0);
insert into order_details values (10546, 35, 18, 30, 0);
insert into order_details values (10546, 62, 49.2999992, 40, 0);
insert into order_details values (10547, 32, 32, 24, 0.150000006);
insert into order_details values (10547, 36, 19, 60, 0);
insert into order_details values (10548, 34, 14, 10, 0.25);
insert into order_details values (10548, 41, 9.64999962, 14, 0);
insert into order_details values (10549, 31, 12.5, 55, 0.150000006);
insert into order_details values (10549, 45, 9.5, 100, 0.150000006);
insert into order_details values (10549, 51, 53, 48, 0.150000006);
insert into order_details values (10550, 17, 39, 8, 0.100000001);
insert into order_details values (10550, 19, 9.19999981, 10, 0);
insert into order_details values (10550, 21, 10, 6, 0.100000001);
insert into order_details values (10550, 61, 28.5, 10, 0.100000001);
insert into order_details values (10551, 16, 17.4500008, 40, 0.150000006);
insert into order_details values (10551, 35, 18, 20, 0.150000006);
insert into order_details values (10551, 44, 19.4500008, 40, 0);
insert into order_details values (10552, 69, 36, 18, 0);
insert into order_details values (10552, 75, 7.75, 30, 0);
insert into order_details values (10553, 11, 21, 15, 0);
insert into order_details values (10553, 16, 17.4500008, 14, 0);
insert into order_details values (10553, 22, 21, 24, 0);
insert into order_details values (10553, 31, 12.5, 30, 0);
insert into order_details values (10553, 35, 18, 6, 0);
insert into order_details values (10554, 16, 17.4500008, 30, 0.0500000007);
insert into order_details values (10554, 23, 9, 20, 0.0500000007);
insert into order_details values (10554, 62, 49.2999992, 20, 0.0500000007);
insert into order_details values (10554, 77, 13, 10, 0.0500000007);
insert into order_details values (10555, 14, 23.25, 30, 0.200000003);
insert into order_details values (10555, 19, 9.19999981, 35, 0.200000003);
insert into order_details values (10555, 24, 4.5, 18, 0.200000003);
insert into order_details values (10555, 51, 53, 20, 0.200000003);
insert into order_details values (10555, 56, 38, 40, 0.200000003);
insert into order_details values (10556, 72, 34.7999992, 24, 0);
insert into order_details values (10557, 64, 33.25, 30, 0);
insert into order_details values (10557, 75, 7.75, 20, 0);
insert into order_details values (10558, 47, 9.5, 25, 0);
insert into order_details values (10558, 51, 53, 20, 0);
insert into order_details values (10558, 52, 7, 30, 0);
insert into order_details values (10558, 53, 32.7999992, 18, 0);
insert into order_details values (10558, 73, 15, 3, 0);
insert into order_details values (10559, 41, 9.64999962, 12, 0.0500000007);
insert into order_details values (10559, 55, 24, 18, 0.0500000007);
insert into order_details values (10560, 30, 25.8899994, 20, 0);
insert into order_details values (10560, 62, 49.2999992, 15, 0.25);
insert into order_details values (10561, 44, 19.4500008, 10, 0);
insert into order_details values (10561, 51, 53, 50, 0);
insert into order_details values (10562, 33, 2.5, 20, 0.100000001);
insert into order_details values (10562, 62, 49.2999992, 10, 0.100000001);
insert into order_details values (10563, 36, 19, 25, 0);
insert into order_details values (10563, 52, 7, 70, 0);
insert into order_details values (10564, 17, 39, 16, 0.0500000007);
insert into order_details values (10564, 31, 12.5, 6, 0.0500000007);
insert into order_details values (10564, 55, 24, 25, 0.0500000007);
insert into order_details values (10565, 24, 4.5, 25, 0.100000001);
insert into order_details values (10565, 64, 33.25, 18, 0.100000001);
insert into order_details values (10566, 11, 21, 35, 0.150000006);
insert into order_details values (10566, 18, 62.5, 18, 0.150000006);
insert into order_details values (10566, 76, 18, 10, 0);
insert into order_details values (10567, 31, 12.5, 60, 0.200000003);
insert into order_details values (10567, 51, 53, 3, 0);
insert into order_details values (10567, 59, 55, 40, 0.200000003);
insert into order_details values (10568, 10, 31, 5, 0);
insert into order_details values (10569, 31, 12.5, 35, 0.200000003);
insert into order_details values (10569, 76, 18, 30, 0);
insert into order_details values (10570, 11, 21, 15, 0.0500000007);
insert into order_details values (10570, 56, 38, 60, 0.0500000007);
insert into order_details values (10571, 14, 23.25, 11, 0.150000006);
insert into order_details values (10571, 42, 14, 28, 0.150000006);
insert into order_details values (10572, 16, 17.4500008, 12, 0.100000001);
insert into order_details values (10572, 32, 32, 10, 0.100000001);
insert into order_details values (10572, 40, 18.3999996, 50, 0);
insert into order_details values (10572, 75, 7.75, 15, 0.100000001);
insert into order_details values (10573, 17, 39, 18, 0);
insert into order_details values (10573, 34, 14, 40, 0);
insert into order_details values (10573, 53, 32.7999992, 25, 0);
insert into order_details values (10574, 33, 2.5, 14, 0);
insert into order_details values (10574, 40, 18.3999996, 2, 0);
insert into order_details values (10574, 62, 49.2999992, 10, 0);
insert into order_details values (10574, 64, 33.25, 6, 0);
insert into order_details values (10575, 59, 55, 12, 0);
insert into order_details values (10575, 63, 43.9000015, 6, 0);
insert into order_details values (10575, 72, 34.7999992, 30, 0);
insert into order_details values (10575, 76, 18, 10, 0);
insert into order_details values (10576, 1, 18, 10, 0);
insert into order_details values (10576, 31, 12.5, 20, 0);
insert into order_details values (10576, 44, 19.4500008, 21, 0);
insert into order_details values (10577, 39, 18, 10, 0);
insert into order_details values (10577, 75, 7.75, 20, 0);
insert into order_details values (10577, 77, 13, 18, 0);
insert into order_details values (10578, 35, 18, 20, 0);
insert into order_details values (10578, 57, 19.5, 6, 0);
insert into order_details values (10579, 15, 15.5, 10, 0);
insert into order_details values (10579, 75, 7.75, 21, 0);
insert into order_details values (10580, 14, 23.25, 15, 0.0500000007);
insert into order_details values (10580, 41, 9.64999962, 9, 0.0500000007);
insert into order_details values (10580, 65, 21.0499992, 30, 0.0500000007);
insert into order_details values (10581, 75, 7.75, 50, 0.200000003);
insert into order_details values (10582, 57, 19.5, 4, 0);
insert into order_details values (10582, 76, 18, 14, 0);
insert into order_details values (10583, 29, 123.790001, 10, 0);
insert into order_details values (10583, 60, 34, 24, 0.150000006);
insert into order_details values (10583, 69, 36, 10, 0.150000006);
insert into order_details values (10584, 31, 12.5, 50, 0.0500000007);
insert into order_details values (10585, 47, 9.5, 15, 0);
insert into order_details values (10586, 52, 7, 4, 0.150000006);
insert into order_details values (10587, 26, 31.2299995, 6, 0);
insert into order_details values (10587, 35, 18, 20, 0);
insert into order_details values (10587, 77, 13, 20, 0);
insert into order_details values (10588, 18, 62.5, 40, 0.200000003);
insert into order_details values (10588, 42, 14, 100, 0.200000003);
insert into order_details values (10589, 35, 18, 4, 0);
insert into order_details values (10590, 1, 18, 20, 0);
insert into order_details values (10590, 77, 13, 60, 0.0500000007);
insert into order_details values (10591, 3, 10, 14, 0);
insert into order_details values (10591, 7, 30, 10, 0);
insert into order_details values (10591, 54, 7.44999981, 50, 0);
insert into order_details values (10592, 15, 15.5, 25, 0.0500000007);
insert into order_details values (10592, 26, 31.2299995, 5, 0.0500000007);
insert into order_details values (10593, 20, 81, 21, 0.200000003);
insert into order_details values (10593, 69, 36, 20, 0.200000003);
insert into order_details values (10593, 76, 18, 4, 0.200000003);
insert into order_details values (10594, 52, 7, 24, 0);
insert into order_details values (10594, 58, 13.25, 30, 0);
insert into order_details values (10595, 35, 18, 30, 0.25);
insert into order_details values (10595, 61, 28.5, 120, 0.25);
insert into order_details values (10595, 69, 36, 65, 0.25);
insert into order_details values (10596, 56, 38, 5, 0.200000003);
insert into order_details values (10596, 63, 43.9000015, 24, 0.200000003);
insert into order_details values (10596, 75, 7.75, 30, 0.200000003);
insert into order_details values (10597, 24, 4.5, 35, 0.200000003);
insert into order_details values (10597, 57, 19.5, 20, 0);
insert into order_details values (10597, 65, 21.0499992, 12, 0.200000003);
insert into order_details values (10598, 27, 43.9000015, 50, 0);
insert into order_details values (10598, 71, 21.5, 9, 0);
insert into order_details values (10599, 62, 49.2999992, 10, 0);
insert into order_details values (10600, 54, 7.44999981, 4, 0);
insert into order_details values (10600, 73, 15, 30, 0);
insert into order_details values (10601, 13, 6, 60, 0);
insert into order_details values (10601, 59, 55, 35, 0);
insert into order_details values (10602, 77, 13, 5, 0.25);
insert into order_details values (10603, 22, 21, 48, 0);
insert into order_details values (10603, 49, 20, 25, 0.0500000007);
insert into order_details values (10604, 48, 12.75, 6, 0.100000001);
insert into order_details values (10604, 76, 18, 10, 0.100000001);
insert into order_details values (10605, 16, 17.4500008, 30, 0.0500000007);
insert into order_details values (10605, 59, 55, 20, 0.0500000007);
insert into order_details values (10605, 60, 34, 70, 0.0500000007);
insert into order_details values (10605, 71, 21.5, 15, 0.0500000007);
insert into order_details values (10606, 4, 22, 20, 0.200000003);
insert into order_details values (10606, 55, 24, 20, 0.200000003);
insert into order_details values (10606, 62, 49.2999992, 10, 0.200000003);
insert into order_details values (10607, 7, 30, 45, 0);
insert into order_details values (10607, 17, 39, 100, 0);
insert into order_details values (10607, 33, 2.5, 14, 0);
insert into order_details values (10607, 40, 18.3999996, 42, 0);
insert into order_details values (10607, 72, 34.7999992, 12, 0);
insert into order_details values (10608, 56, 38, 28, 0);
insert into order_details values (10609, 1, 18, 3, 0);
insert into order_details values (10609, 10, 31, 10, 0);
insert into order_details values (10609, 21, 10, 6, 0);
insert into order_details values (10610, 36, 19, 21, 0.25);
insert into order_details values (10611, 1, 18, 6, 0);
insert into order_details values (10611, 2, 19, 10, 0);
insert into order_details values (10611, 60, 34, 15, 0);
insert into order_details values (10612, 10, 31, 70, 0);
insert into order_details values (10612, 36, 19, 55, 0);
insert into order_details values (10612, 49, 20, 18, 0);
insert into order_details values (10612, 60, 34, 40, 0);
insert into order_details values (10612, 76, 18, 80, 0);
insert into order_details values (10613, 13, 6, 8, 0.100000001);
insert into order_details values (10613, 75, 7.75, 40, 0);
insert into order_details values (10614, 11, 21, 14, 0);
insert into order_details values (10614, 21, 10, 8, 0);
insert into order_details values (10614, 39, 18, 5, 0);
insert into order_details values (10615, 55, 24, 5, 0);
insert into order_details values (10616, 38, 263.5, 15, 0.0500000007);
insert into order_details values (10616, 56, 38, 14, 0);
insert into order_details values (10616, 70, 15, 15, 0.0500000007);
insert into order_details values (10616, 71, 21.5, 15, 0.0500000007);
insert into order_details values (10617, 59, 55, 30, 0.150000006);
insert into order_details values (10618, 6, 25, 70, 0);
insert into order_details values (10618, 56, 38, 20, 0);
insert into order_details values (10618, 68, 12.5, 15, 0);
insert into order_details values (10619, 21, 10, 42, 0);
insert into order_details values (10619, 22, 21, 40, 0);
insert into order_details values (10620, 24, 4.5, 5, 0);
insert into order_details values (10620, 52, 7, 5, 0);
insert into order_details values (10621, 19, 9.19999981, 5, 0);
insert into order_details values (10621, 23, 9, 10, 0);
insert into order_details values (10621, 70, 15, 20, 0);
insert into order_details values (10621, 71, 21.5, 15, 0);
insert into order_details values (10622, 2, 19, 20, 0);
insert into order_details values (10622, 68, 12.5, 18, 0.200000003);
insert into order_details values (10623, 14, 23.25, 21, 0);
insert into order_details values (10623, 19, 9.19999981, 15, 0.100000001);
insert into order_details values (10623, 21, 10, 25, 0.100000001);
insert into order_details values (10623, 24, 4.5, 3, 0);
insert into order_details values (10623, 35, 18, 30, 0.100000001);
insert into order_details values (10624, 28, 45.5999985, 10, 0);
insert into order_details values (10624, 29, 123.790001, 6, 0);
insert into order_details values (10624, 44, 19.4500008, 10, 0);
insert into order_details values (10625, 14, 23.25, 3, 0);
insert into order_details values (10625, 42, 14, 5, 0);
insert into order_details values (10625, 60, 34, 10, 0);
insert into order_details values (10626, 53, 32.7999992, 12, 0);
insert into order_details values (10626, 60, 34, 20, 0);
insert into order_details values (10626, 71, 21.5, 20, 0);
insert into order_details values (10627, 62, 49.2999992, 15, 0);
insert into order_details values (10627, 73, 15, 35, 0.150000006);
insert into order_details values (10628, 1, 18, 25, 0);
insert into order_details values (10629, 29, 123.790001, 20, 0);
insert into order_details values (10629, 64, 33.25, 9, 0);
insert into order_details values (10630, 55, 24, 12, 0.0500000007);
insert into order_details values (10630, 76, 18, 35, 0);
insert into order_details values (10631, 75, 7.75, 8, 0.100000001);
insert into order_details values (10632, 2, 19, 30, 0.0500000007);
insert into order_details values (10632, 33, 2.5, 20, 0.0500000007);
insert into order_details values (10633, 12, 38, 36, 0.150000006);
insert into order_details values (10633, 13, 6, 13, 0.150000006);
insert into order_details values (10633, 26, 31.2299995, 35, 0.150000006);
insert into order_details values (10633, 62, 49.2999992, 80, 0.150000006);
insert into order_details values (10634, 7, 30, 35, 0);
insert into order_details values (10634, 18, 62.5, 50, 0);
insert into order_details values (10634, 51, 53, 15, 0);
insert into order_details values (10634, 75, 7.75, 2, 0);
insert into order_details values (10635, 4, 22, 10, 0.100000001);
insert into order_details values (10635, 5, 21.3500004, 15, 0.100000001);
insert into order_details values (10635, 22, 21, 40, 0);
insert into order_details values (10636, 4, 22, 25, 0);
insert into order_details values (10636, 58, 13.25, 6, 0);
insert into order_details values (10637, 11, 21, 10, 0);
insert into order_details values (10637, 50, 16.25, 25, 0.0500000007);
insert into order_details values (10637, 56, 38, 60, 0.0500000007);
insert into order_details values (10638, 45, 9.5, 20, 0);
insert into order_details values (10638, 65, 21.0499992, 21, 0);
insert into order_details values (10638, 72, 34.7999992, 60, 0);
insert into order_details values (10639, 18, 62.5, 8, 0);
insert into order_details values (10640, 69, 36, 20, 0.25);
insert into order_details values (10640, 70, 15, 15, 0.25);
insert into order_details values (10641, 2, 19, 50, 0);
insert into order_details values (10641, 40, 18.3999996, 60, 0);
insert into order_details values (10642, 21, 10, 30, 0.200000003);
insert into order_details values (10642, 61, 28.5, 20, 0.200000003);
insert into order_details values (10643, 28, 45.5999985, 15, 0.25);
insert into order_details values (10643, 39, 18, 21, 0.25);
insert into order_details values (10643, 46, 12, 2, 0.25);
insert into order_details values (10644, 18, 62.5, 4, 0.100000001);
insert into order_details values (10644, 43, 46, 20, 0);
insert into order_details values (10644, 46, 12, 21, 0.100000001);
insert into order_details values (10645, 18, 62.5, 20, 0);
insert into order_details values (10645, 36, 19, 15, 0);
insert into order_details values (10646, 1, 18, 15, 0.25);
insert into order_details values (10646, 10, 31, 18, 0.25);
insert into order_details values (10646, 71, 21.5, 30, 0.25);
insert into order_details values (10646, 77, 13, 35, 0.25);
insert into order_details values (10647, 19, 9.19999981, 30, 0);
insert into order_details values (10647, 39, 18, 20, 0);
insert into order_details values (10648, 22, 21, 15, 0);
insert into order_details values (10648, 24, 4.5, 15, 0.150000006);
insert into order_details values (10649, 28, 45.5999985, 20, 0);
insert into order_details values (10649, 72, 34.7999992, 15, 0);
insert into order_details values (10650, 30, 25.8899994, 30, 0);
insert into order_details values (10650, 53, 32.7999992, 25, 0.0500000007);
insert into order_details values (10650, 54, 7.44999981, 30, 0);
insert into order_details values (10651, 19, 9.19999981, 12, 0.25);
insert into order_details values (10651, 22, 21, 20, 0.25);
insert into order_details values (10652, 30, 25.8899994, 2, 0.25);
insert into order_details values (10652, 42, 14, 20, 0);
insert into order_details values (10653, 16, 17.4500008, 30, 0.100000001);
insert into order_details values (10653, 60, 34, 20, 0.100000001);
insert into order_details values (10654, 4, 22, 12, 0.100000001);
insert into order_details values (10654, 39, 18, 20, 0.100000001);
insert into order_details values (10654, 54, 7.44999981, 6, 0.100000001);
insert into order_details values (10655, 41, 9.64999962, 20, 0.200000003);
insert into order_details values (10656, 14, 23.25, 3, 0.100000001);
insert into order_details values (10656, 44, 19.4500008, 28, 0.100000001);
insert into order_details values (10656, 47, 9.5, 6, 0.100000001);
insert into order_details values (10657, 15, 15.5, 50, 0);
insert into order_details values (10657, 41, 9.64999962, 24, 0);
insert into order_details values (10657, 46, 12, 45, 0);
insert into order_details values (10657, 47, 9.5, 10, 0);
insert into order_details values (10657, 56, 38, 45, 0);
insert into order_details values (10657, 60, 34, 30, 0);
insert into order_details values (10658, 21, 10, 60, 0);
insert into order_details values (10658, 40, 18.3999996, 70, 0.0500000007);
insert into order_details values (10658, 60, 34, 55, 0.0500000007);
insert into order_details values (10658, 77, 13, 70, 0.0500000007);
insert into order_details values (10659, 31, 12.5, 20, 0.0500000007);
insert into order_details values (10659, 40, 18.3999996, 24, 0.0500000007);
insert into order_details values (10659, 70, 15, 40, 0.0500000007);
insert into order_details values (10660, 20, 81, 21, 0);
insert into order_details values (10661, 39, 18, 3, 0.200000003);
insert into order_details values (10661, 58, 13.25, 49, 0.200000003);
insert into order_details values (10662, 68, 12.5, 10, 0);
insert into order_details values (10663, 40, 18.3999996, 30, 0.0500000007);
insert into order_details values (10663, 42, 14, 30, 0.0500000007);
insert into order_details values (10663, 51, 53, 20, 0.0500000007);
insert into order_details values (10664, 10, 31, 24, 0.150000006);
insert into order_details values (10664, 56, 38, 12, 0.150000006);
insert into order_details values (10664, 65, 21.0499992, 15, 0.150000006);
insert into order_details values (10665, 51, 53, 20, 0);
insert into order_details values (10665, 59, 55, 1, 0);
insert into order_details values (10665, 76, 18, 10, 0);
insert into order_details values (10666, 29, 123.790001, 36, 0);
insert into order_details values (10666, 65, 21.0499992, 10, 0);
insert into order_details values (10667, 69, 36, 45, 0.200000003);
insert into order_details values (10667, 71, 21.5, 14, 0.200000003);
insert into order_details values (10668, 31, 12.5, 8, 0.100000001);
insert into order_details values (10668, 55, 24, 4, 0.100000001);
insert into order_details values (10668, 64, 33.25, 15, 0.100000001);
insert into order_details values (10669, 36, 19, 30, 0);
insert into order_details values (10670, 23, 9, 32, 0);
insert into order_details values (10670, 46, 12, 60, 0);
insert into order_details values (10670, 67, 14, 25, 0);
insert into order_details values (10670, 73, 15, 50, 0);
insert into order_details values (10670, 75, 7.75, 25, 0);
insert into order_details values (10671, 16, 17.4500008, 10, 0);
insert into order_details values (10671, 62, 49.2999992, 10, 0);
insert into order_details values (10671, 65, 21.0499992, 12, 0);
insert into order_details values (10672, 38, 263.5, 15, 0.100000001);
insert into order_details values (10672, 71, 21.5, 12, 0);
insert into order_details values (10673, 16, 17.4500008, 3, 0);
insert into order_details values (10673, 42, 14, 6, 0);
insert into order_details values (10673, 43, 46, 6, 0);
insert into order_details values (10674, 23, 9, 5, 0);
insert into order_details values (10675, 14, 23.25, 30, 0);
insert into order_details values (10675, 53, 32.7999992, 10, 0);
insert into order_details values (10675, 58, 13.25, 30, 0);
insert into order_details values (10676, 10, 31, 2, 0);
insert into order_details values (10676, 19, 9.19999981, 7, 0);
insert into order_details values (10676, 44, 19.4500008, 21, 0);
insert into order_details values (10677, 26, 31.2299995, 30, 0.150000006);
insert into order_details values (10677, 33, 2.5, 8, 0.150000006);
insert into order_details values (10678, 12, 38, 100, 0);
insert into order_details values (10678, 33, 2.5, 30, 0);
insert into order_details values (10678, 41, 9.64999962, 120, 0);
insert into order_details values (10678, 54, 7.44999981, 30, 0);
insert into order_details values (10679, 59, 55, 12, 0);
insert into order_details values (10680, 16, 17.4500008, 50, 0.25);
insert into order_details values (10680, 31, 12.5, 20, 0.25);
insert into order_details values (10680, 42, 14, 40, 0.25);
insert into order_details values (10681, 19, 9.19999981, 30, 0.100000001);
insert into order_details values (10681, 21, 10, 12, 0.100000001);
insert into order_details values (10681, 64, 33.25, 28, 0);
insert into order_details values (10682, 33, 2.5, 30, 0);
insert into order_details values (10682, 66, 17, 4, 0);
insert into order_details values (10682, 75, 7.75, 30, 0);
insert into order_details values (10683, 52, 7, 9, 0);
insert into order_details values (10684, 40, 18.3999996, 20, 0);
insert into order_details values (10684, 47, 9.5, 40, 0);
insert into order_details values (10684, 60, 34, 30, 0);
insert into order_details values (10685, 10, 31, 20, 0);
insert into order_details values (10685, 41, 9.64999962, 4, 0);
insert into order_details values (10685, 47, 9.5, 15, 0);
insert into order_details values (10686, 17, 39, 30, 0.200000003);
insert into order_details values (10686, 26, 31.2299995, 15, 0);
insert into order_details values (10687, 9, 97, 50, 0.25);
insert into order_details values (10687, 29, 123.790001, 10, 0);
insert into order_details values (10687, 36, 19, 6, 0.25);
insert into order_details values (10688, 10, 31, 18, 0.100000001);
insert into order_details values (10688, 28, 45.5999985, 60, 0.100000001);
insert into order_details values (10688, 34, 14, 14, 0);
insert into order_details values (10689, 1, 18, 35, 0.25);
insert into order_details values (10690, 56, 38, 20, 0.25);
insert into order_details values (10690, 77, 13, 30, 0.25);
insert into order_details values (10691, 1, 18, 30, 0);
insert into order_details values (10691, 29, 123.790001, 40, 0);
insert into order_details values (10691, 43, 46, 40, 0);
insert into order_details values (10691, 44, 19.4500008, 24, 0);
insert into order_details values (10691, 62, 49.2999992, 48, 0);
insert into order_details values (10692, 63, 43.9000015, 20, 0);
insert into order_details values (10693, 9, 97, 6, 0);
insert into order_details values (10693, 54, 7.44999981, 60, 0.150000006);
insert into order_details values (10693, 69, 36, 30, 0.150000006);
insert into order_details values (10693, 73, 15, 15, 0.150000006);
insert into order_details values (10694, 7, 30, 90, 0);
insert into order_details values (10694, 59, 55, 25, 0);
insert into order_details values (10694, 70, 15, 50, 0);
insert into order_details values (10695, 8, 40, 10, 0);
insert into order_details values (10695, 12, 38, 4, 0);
insert into order_details values (10695, 24, 4.5, 20, 0);
insert into order_details values (10696, 17, 39, 20, 0);
insert into order_details values (10696, 46, 12, 18, 0);
insert into order_details values (10697, 19, 9.19999981, 7, 0.25);
insert into order_details values (10697, 35, 18, 9, 0.25);
insert into order_details values (10697, 58, 13.25, 30, 0.25);
insert into order_details values (10697, 70, 15, 30, 0.25);
insert into order_details values (10698, 11, 21, 15, 0);
insert into order_details values (10698, 17, 39, 8, 0.0500000007);
insert into order_details values (10698, 29, 123.790001, 12, 0.0500000007);
insert into order_details values (10698, 65, 21.0499992, 65, 0.0500000007);
insert into order_details values (10698, 70, 15, 8, 0.0500000007);
insert into order_details values (10699, 47, 9.5, 12, 0);
insert into order_details values (10700, 1, 18, 5, 0.200000003);
insert into order_details values (10700, 34, 14, 12, 0.200000003);
insert into order_details values (10700, 68, 12.5, 40, 0.200000003);
insert into order_details values (10700, 71, 21.5, 60, 0.200000003);
insert into order_details values (10701, 59, 55, 42, 0.150000006);
insert into order_details values (10701, 71, 21.5, 20, 0.150000006);
insert into order_details values (10701, 76, 18, 35, 0.150000006);
insert into order_details values (10702, 3, 10, 6, 0);
insert into order_details values (10702, 76, 18, 15, 0);
insert into order_details values (10703, 2, 19, 5, 0);
insert into order_details values (10703, 59, 55, 35, 0);
insert into order_details values (10703, 73, 15, 35, 0);
insert into order_details values (10704, 4, 22, 6, 0);
insert into order_details values (10704, 24, 4.5, 35, 0);
insert into order_details values (10704, 48, 12.75, 24, 0);
insert into order_details values (10705, 31, 12.5, 20, 0);
insert into order_details values (10705, 32, 32, 4, 0);
insert into order_details values (10706, 16, 17.4500008, 20, 0);
insert into order_details values (10706, 43, 46, 24, 0);
insert into order_details values (10706, 59, 55, 8, 0);
insert into order_details values (10707, 55, 24, 21, 0);
insert into order_details values (10707, 57, 19.5, 40, 0);
insert into order_details values (10707, 70, 15, 28, 0.150000006);
insert into order_details values (10708, 5, 21.3500004, 4, 0);
insert into order_details values (10708, 36, 19, 5, 0);
insert into order_details values (10709, 8, 40, 40, 0);
insert into order_details values (10709, 51, 53, 28, 0);
insert into order_details values (10709, 60, 34, 10, 0);
insert into order_details values (10710, 19, 9.19999981, 5, 0);
insert into order_details values (10710, 47, 9.5, 5, 0);
insert into order_details values (10711, 19, 9.19999981, 12, 0);
insert into order_details values (10711, 41, 9.64999962, 42, 0);
insert into order_details values (10711, 53, 32.7999992, 120, 0);
insert into order_details values (10712, 53, 32.7999992, 3, 0.0500000007);
insert into order_details values (10712, 56, 38, 30, 0);
insert into order_details values (10713, 10, 31, 18, 0);
insert into order_details values (10713, 26, 31.2299995, 30, 0);
insert into order_details values (10713, 45, 9.5, 110, 0);
insert into order_details values (10713, 46, 12, 24, 0);
insert into order_details values (10714, 2, 19, 30, 0.25);
insert into order_details values (10714, 17, 39, 27, 0.25);
insert into order_details values (10714, 47, 9.5, 50, 0.25);
insert into order_details values (10714, 56, 38, 18, 0.25);
insert into order_details values (10714, 58, 13.25, 12, 0.25);
insert into order_details values (10715, 10, 31, 21, 0);
insert into order_details values (10715, 71, 21.5, 30, 0);
insert into order_details values (10716, 21, 10, 5, 0);
insert into order_details values (10716, 51, 53, 7, 0);
insert into order_details values (10716, 61, 28.5, 10, 0);
insert into order_details values (10717, 21, 10, 32, 0.0500000007);
insert into order_details values (10717, 54, 7.44999981, 15, 0);
insert into order_details values (10717, 69, 36, 25, 0.0500000007);
insert into order_details values (10718, 12, 38, 36, 0);
insert into order_details values (10718, 16, 17.4500008, 20, 0);
insert into order_details values (10718, 36, 19, 40, 0);
insert into order_details values (10718, 62, 49.2999992, 20, 0);
insert into order_details values (10719, 18, 62.5, 12, 0.25);
insert into order_details values (10719, 30, 25.8899994, 3, 0.25);
insert into order_details values (10719, 54, 7.44999981, 40, 0.25);
insert into order_details values (10720, 35, 18, 21, 0);
insert into order_details values (10720, 71, 21.5, 8, 0);
insert into order_details values (10721, 44, 19.4500008, 50, 0.0500000007);
insert into order_details values (10722, 2, 19, 3, 0);
insert into order_details values (10722, 31, 12.5, 50, 0);
insert into order_details values (10722, 68, 12.5, 45, 0);
insert into order_details values (10722, 75, 7.75, 42, 0);
insert into order_details values (10723, 26, 31.2299995, 15, 0);
insert into order_details values (10724, 10, 31, 16, 0);
insert into order_details values (10724, 61, 28.5, 5, 0);
insert into order_details values (10725, 41, 9.64999962, 12, 0);
insert into order_details values (10725, 52, 7, 4, 0);
insert into order_details values (10725, 55, 24, 6, 0);
insert into order_details values (10726, 4, 22, 25, 0);
insert into order_details values (10726, 11, 21, 5, 0);
insert into order_details values (10727, 17, 39, 20, 0.0500000007);
insert into order_details values (10727, 56, 38, 10, 0.0500000007);
insert into order_details values (10727, 59, 55, 10, 0.0500000007);
insert into order_details values (10728, 30, 25.8899994, 15, 0);
insert into order_details values (10728, 40, 18.3999996, 6, 0);
insert into order_details values (10728, 55, 24, 12, 0);
insert into order_details values (10728, 60, 34, 15, 0);
insert into order_details values (10729, 1, 18, 50, 0);
insert into order_details values (10729, 21, 10, 30, 0);
insert into order_details values (10729, 50, 16.25, 40, 0);
insert into order_details values (10730, 16, 17.4500008, 15, 0.0500000007);
insert into order_details values (10730, 31, 12.5, 3, 0.0500000007);
insert into order_details values (10730, 65, 21.0499992, 10, 0.0500000007);
insert into order_details values (10731, 21, 10, 40, 0.0500000007);
insert into order_details values (10731, 51, 53, 30, 0.0500000007);
insert into order_details values (10732, 76, 18, 20, 0);
insert into order_details values (10733, 14, 23.25, 16, 0);
insert into order_details values (10733, 28, 45.5999985, 20, 0);
insert into order_details values (10733, 52, 7, 25, 0);
insert into order_details values (10734, 6, 25, 30, 0);
insert into order_details values (10734, 30, 25.8899994, 15, 0);
insert into order_details values (10734, 76, 18, 20, 0);
insert into order_details values (10735, 61, 28.5, 20, 0.100000001);
insert into order_details values (10735, 77, 13, 2, 0.100000001);
insert into order_details values (10736, 65, 21.0499992, 40, 0);
insert into order_details values (10736, 75, 7.75, 20, 0);
insert into order_details values (10737, 13, 6, 4, 0);
insert into order_details values (10737, 41, 9.64999962, 12, 0);
insert into order_details values (10738, 16, 17.4500008, 3, 0);
insert into order_details values (10739, 36, 19, 6, 0);
insert into order_details values (10739, 52, 7, 18, 0);
insert into order_details values (10740, 28, 45.5999985, 5, 0.200000003);
insert into order_details values (10740, 35, 18, 35, 0.200000003);
insert into order_details values (10740, 45, 9.5, 40, 0.200000003);
insert into order_details values (10740, 56, 38, 14, 0.200000003);
insert into order_details values (10741, 2, 19, 15, 0.200000003);
insert into order_details values (10742, 3, 10, 20, 0);
insert into order_details values (10742, 60, 34, 50, 0);
insert into order_details values (10742, 72, 34.7999992, 35, 0);
insert into order_details values (10743, 46, 12, 28, 0.0500000007);
insert into order_details values (10744, 40, 18.3999996, 50, 0.200000003);
insert into order_details values (10745, 18, 62.5, 24, 0);
insert into order_details values (10745, 44, 19.4500008, 16, 0);
insert into order_details values (10745, 59, 55, 45, 0);
insert into order_details values (10745, 72, 34.7999992, 7, 0);
insert into order_details values (10746, 13, 6, 6, 0);
insert into order_details values (10746, 42, 14, 28, 0);
insert into order_details values (10746, 62, 49.2999992, 9, 0);
insert into order_details values (10746, 69, 36, 40, 0);
insert into order_details values (10747, 31, 12.5, 8, 0);
insert into order_details values (10747, 41, 9.64999962, 35, 0);
insert into order_details values (10747, 63, 43.9000015, 9, 0);
insert into order_details values (10747, 69, 36, 30, 0);
insert into order_details values (10748, 23, 9, 44, 0);
insert into order_details values (10748, 40, 18.3999996, 40, 0);
insert into order_details values (10748, 56, 38, 28, 0);
insert into order_details values (10749, 56, 38, 15, 0);
insert into order_details values (10749, 59, 55, 6, 0);
insert into order_details values (10749, 76, 18, 10, 0);
insert into order_details values (10750, 14, 23.25, 5, 0.150000006);
insert into order_details values (10750, 45, 9.5, 40, 0.150000006);
insert into order_details values (10750, 59, 55, 25, 0.150000006);
insert into order_details values (10751, 26, 31.2299995, 12, 0.100000001);
insert into order_details values (10751, 30, 25.8899994, 30, 0);
insert into order_details values (10751, 50, 16.25, 20, 0.100000001);
insert into order_details values (10751, 73, 15, 15, 0);
insert into order_details values (10752, 1, 18, 8, 0);
insert into order_details values (10752, 69, 36, 3, 0);
insert into order_details values (10753, 45, 9.5, 4, 0);
insert into order_details values (10753, 74, 10, 5, 0);
insert into order_details values (10754, 40, 18.3999996, 3, 0);
insert into order_details values (10755, 47, 9.5, 30, 0.25);
insert into order_details values (10755, 56, 38, 30, 0.25);
insert into order_details values (10755, 57, 19.5, 14, 0.25);
insert into order_details values (10755, 69, 36, 25, 0.25);
insert into order_details values (10756, 18, 62.5, 21, 0.200000003);
insert into order_details values (10756, 36, 19, 20, 0.200000003);
insert into order_details values (10756, 68, 12.5, 6, 0.200000003);
insert into order_details values (10756, 69, 36, 20, 0.200000003);
insert into order_details values (10757, 34, 14, 30, 0);
insert into order_details values (10757, 59, 55, 7, 0);
insert into order_details values (10757, 62, 49.2999992, 30, 0);
insert into order_details values (10757, 64, 33.25, 24, 0);
insert into order_details values (10758, 26, 31.2299995, 20, 0);
insert into order_details values (10758, 52, 7, 60, 0);
insert into order_details values (10758, 70, 15, 40, 0);
insert into order_details values (10759, 32, 32, 10, 0);
insert into order_details values (10760, 25, 14, 12, 0.25);
insert into order_details values (10760, 27, 43.9000015, 40, 0);
insert into order_details values (10760, 43, 46, 30, 0.25);
insert into order_details values (10761, 25, 14, 35, 0.25);
insert into order_details values (10761, 75, 7.75, 18, 0);
insert into order_details values (10762, 39, 18, 16, 0);
insert into order_details values (10762, 47, 9.5, 30, 0);
insert into order_details values (10762, 51, 53, 28, 0);
insert into order_details values (10762, 56, 38, 60, 0);
insert into order_details values (10763, 21, 10, 40, 0);
insert into order_details values (10763, 22, 21, 6, 0);
insert into order_details values (10763, 24, 4.5, 20, 0);
insert into order_details values (10764, 3, 10, 20, 0.100000001);
insert into order_details values (10764, 39, 18, 130, 0.100000001);
insert into order_details values (10765, 65, 21.0499992, 80, 0.100000001);
insert into order_details values (10766, 2, 19, 40, 0);
insert into order_details values (10766, 7, 30, 35, 0);
insert into order_details values (10766, 68, 12.5, 40, 0);
insert into order_details values (10767, 42, 14, 2, 0);
insert into order_details values (10768, 22, 21, 4, 0);
insert into order_details values (10768, 31, 12.5, 50, 0);
insert into order_details values (10768, 60, 34, 15, 0);
insert into order_details values (10768, 71, 21.5, 12, 0);
insert into order_details values (10769, 41, 9.64999962, 30, 0.0500000007);
insert into order_details values (10769, 52, 7, 15, 0.0500000007);
insert into order_details values (10769, 61, 28.5, 20, 0);
insert into order_details values (10769, 62, 49.2999992, 15, 0);
insert into order_details values (10770, 11, 21, 15, 0.25);
insert into order_details values (10771, 71, 21.5, 16, 0);
insert into order_details values (10772, 29, 123.790001, 18, 0);
insert into order_details values (10772, 59, 55, 25, 0);
insert into order_details values (10773, 17, 39, 33, 0);
insert into order_details values (10773, 31, 12.5, 70, 0.200000003);
insert into order_details values (10773, 75, 7.75, 7, 0.200000003);
insert into order_details values (10774, 31, 12.5, 2, 0.25);
insert into order_details values (10774, 66, 17, 50, 0);
insert into order_details values (10775, 10, 31, 6, 0);
insert into order_details values (10775, 67, 14, 3, 0);
insert into order_details values (10776, 31, 12.5, 16, 0.0500000007);
insert into order_details values (10776, 42, 14, 12, 0.0500000007);
insert into order_details values (10776, 45, 9.5, 27, 0.0500000007);
insert into order_details values (10776, 51, 53, 120, 0.0500000007);
insert into order_details values (10777, 42, 14, 20, 0.200000003);
insert into order_details values (10778, 41, 9.64999962, 10, 0);
insert into order_details values (10779, 16, 17.4500008, 20, 0);
insert into order_details values (10779, 62, 49.2999992, 20, 0);
insert into order_details values (10780, 70, 15, 35, 0);
insert into order_details values (10780, 77, 13, 15, 0);
insert into order_details values (10781, 54, 7.44999981, 3, 0.200000003);
insert into order_details values (10781, 56, 38, 20, 0.200000003);
insert into order_details values (10781, 74, 10, 35, 0);
insert into order_details values (10782, 31, 12.5, 1, 0);
insert into order_details values (10783, 31, 12.5, 10, 0);
insert into order_details values (10783, 38, 263.5, 5, 0);
insert into order_details values (10784, 36, 19, 30, 0);
insert into order_details values (10784, 39, 18, 2, 0.150000006);
insert into order_details values (10784, 72, 34.7999992, 30, 0.150000006);
insert into order_details values (10785, 10, 31, 10, 0);
insert into order_details values (10785, 75, 7.75, 10, 0);
insert into order_details values (10786, 8, 40, 30, 0.200000003);
insert into order_details values (10786, 30, 25.8899994, 15, 0.200000003);
insert into order_details values (10786, 75, 7.75, 42, 0.200000003);
insert into order_details values (10787, 2, 19, 15, 0.0500000007);
insert into order_details values (10787, 29, 123.790001, 20, 0.0500000007);
insert into order_details values (10788, 19, 9.19999981, 50, 0.0500000007);
insert into order_details values (10788, 75, 7.75, 40, 0.0500000007);
insert into order_details values (10789, 18, 62.5, 30, 0);
insert into order_details values (10789, 35, 18, 15, 0);
insert into order_details values (10789, 63, 43.9000015, 30, 0);
insert into order_details values (10789, 68, 12.5, 18, 0);
insert into order_details values (10790, 7, 30, 3, 0.150000006);
insert into order_details values (10790, 56, 38, 20, 0.150000006);
insert into order_details values (10791, 29, 123.790001, 14, 0.0500000007);
insert into order_details values (10791, 41, 9.64999962, 20, 0.0500000007);
insert into order_details values (10792, 2, 19, 10, 0);
insert into order_details values (10792, 54, 7.44999981, 3, 0);
insert into order_details values (10792, 68, 12.5, 15, 0);
insert into order_details values (10793, 41, 9.64999962, 14, 0);
insert into order_details values (10793, 52, 7, 8, 0);
insert into order_details values (10794, 14, 23.25, 15, 0.200000003);
insert into order_details values (10794, 54, 7.44999981, 6, 0.200000003);
insert into order_details values (10795, 16, 17.4500008, 65, 0);
insert into order_details values (10795, 17, 39, 35, 0.25);
insert into order_details values (10796, 26, 31.2299995, 21, 0.200000003);
insert into order_details values (10796, 44, 19.4500008, 10, 0);
insert into order_details values (10796, 64, 33.25, 35, 0.200000003);
insert into order_details values (10796, 69, 36, 24, 0.200000003);
insert into order_details values (10797, 11, 21, 20, 0);
insert into order_details values (10798, 62, 49.2999992, 2, 0);
insert into order_details values (10798, 72, 34.7999992, 10, 0);
insert into order_details values (10799, 13, 6, 20, 0.150000006);
insert into order_details values (10799, 24, 4.5, 20, 0.150000006);
insert into order_details values (10799, 59, 55, 25, 0);
insert into order_details values (10800, 11, 21, 50, 0.100000001);
insert into order_details values (10800, 51, 53, 10, 0.100000001);
insert into order_details values (10800, 54, 7.44999981, 7, 0.100000001);
insert into order_details values (10801, 17, 39, 40, 0.25);
insert into order_details values (10801, 29, 123.790001, 20, 0.25);
insert into order_details values (10802, 30, 25.8899994, 25, 0.25);
insert into order_details values (10802, 51, 53, 30, 0.25);
insert into order_details values (10802, 55, 24, 60, 0.25);
insert into order_details values (10802, 62, 49.2999992, 5, 0.25);
insert into order_details values (10803, 19, 9.19999981, 24, 0.0500000007);
insert into order_details values (10803, 25, 14, 15, 0.0500000007);
insert into order_details values (10803, 59, 55, 15, 0.0500000007);
insert into order_details values (10804, 10, 31, 36, 0);
insert into order_details values (10804, 28, 45.5999985, 24, 0);
insert into order_details values (10804, 49, 20, 4, 0.150000006);
insert into order_details values (10805, 34, 14, 10, 0);
insert into order_details values (10805, 38, 263.5, 10, 0);
insert into order_details values (10806, 2, 19, 20, 0.25);
insert into order_details values (10806, 65, 21.0499992, 2, 0);
insert into order_details values (10806, 74, 10, 15, 0.25);
insert into order_details values (10807, 40, 18.3999996, 1, 0);
insert into order_details values (10808, 56, 38, 20, 0.150000006);
insert into order_details values (10808, 76, 18, 50, 0.150000006);
insert into order_details values (10809, 52, 7, 20, 0);
insert into order_details values (10810, 13, 6, 7, 0);
insert into order_details values (10810, 25, 14, 5, 0);
insert into order_details values (10810, 70, 15, 5, 0);
insert into order_details values (10811, 19, 9.19999981, 15, 0);
insert into order_details values (10811, 23, 9, 18, 0);
insert into order_details values (10811, 40, 18.3999996, 30, 0);
insert into order_details values (10812, 31, 12.5, 16, 0.100000001);
insert into order_details values (10812, 72, 34.7999992, 40, 0.100000001);
insert into order_details values (10812, 77, 13, 20, 0);
insert into order_details values (10813, 2, 19, 12, 0.200000003);
insert into order_details values (10813, 46, 12, 35, 0);
insert into order_details values (10814, 41, 9.64999962, 20, 0);
insert into order_details values (10814, 43, 46, 20, 0.150000006);
insert into order_details values (10814, 48, 12.75, 8, 0.150000006);
insert into order_details values (10814, 61, 28.5, 30, 0.150000006);
insert into order_details values (10815, 33, 2.5, 16, 0);
insert into order_details values (10816, 38, 263.5, 30, 0.0500000007);
insert into order_details values (10816, 62, 49.2999992, 20, 0.0500000007);
insert into order_details values (10817, 26, 31.2299995, 40, 0.150000006);
insert into order_details values (10817, 38, 263.5, 30, 0);
insert into order_details values (10817, 40, 18.3999996, 60, 0.150000006);
insert into order_details values (10817, 62, 49.2999992, 25, 0.150000006);
insert into order_details values (10818, 32, 32, 20, 0);
insert into order_details values (10818, 41, 9.64999962, 20, 0);
insert into order_details values (10819, 43, 46, 7, 0);
insert into order_details values (10819, 75, 7.75, 20, 0);
insert into order_details values (10820, 56, 38, 30, 0);
insert into order_details values (10821, 35, 18, 20, 0);
insert into order_details values (10821, 51, 53, 6, 0);
insert into order_details values (10822, 62, 49.2999992, 3, 0);
insert into order_details values (10822, 70, 15, 6, 0);
insert into order_details values (10823, 11, 21, 20, 0.100000001);
insert into order_details values (10823, 57, 19.5, 15, 0);
insert into order_details values (10823, 59, 55, 40, 0.100000001);
insert into order_details values (10823, 77, 13, 15, 0.100000001);
insert into order_details values (10824, 41, 9.64999962, 12, 0);
insert into order_details values (10824, 70, 15, 9, 0);
insert into order_details values (10825, 26, 31.2299995, 12, 0);
insert into order_details values (10825, 53, 32.7999992, 20, 0);
insert into order_details values (10826, 31, 12.5, 35, 0);
insert into order_details values (10826, 57, 19.5, 15, 0);
insert into order_details values (10827, 10, 31, 15, 0);
insert into order_details values (10827, 39, 18, 21, 0);
insert into order_details values (10828, 20, 81, 5, 0);
insert into order_details values (10828, 38, 263.5, 2, 0);
insert into order_details values (10829, 2, 19, 10, 0);
insert into order_details values (10829, 8, 40, 20, 0);
insert into order_details values (10829, 13, 6, 10, 0);
insert into order_details values (10829, 60, 34, 21, 0);
insert into order_details values (10830, 6, 25, 6, 0);
insert into order_details values (10830, 39, 18, 28, 0);
insert into order_details values (10830, 60, 34, 30, 0);
insert into order_details values (10830, 68, 12.5, 24, 0);
insert into order_details values (10831, 19, 9.19999981, 2, 0);
insert into order_details values (10831, 35, 18, 8, 0);
insert into order_details values (10831, 38, 263.5, 8, 0);
insert into order_details values (10831, 43, 46, 9, 0);
insert into order_details values (10832, 13, 6, 3, 0.200000003);
insert into order_details values (10832, 25, 14, 10, 0.200000003);
insert into order_details values (10832, 44, 19.4500008, 16, 0.200000003);
insert into order_details values (10832, 64, 33.25, 3, 0);
insert into order_details values (10833, 7, 30, 20, 0.100000001);
insert into order_details values (10833, 31, 12.5, 9, 0.100000001);
insert into order_details values (10833, 53, 32.7999992, 9, 0.100000001);
insert into order_details values (10834, 29, 123.790001, 8, 0.0500000007);
insert into order_details values (10834, 30, 25.8899994, 20, 0.0500000007);
insert into order_details values (10835, 59, 55, 15, 0);
insert into order_details values (10835, 77, 13, 2, 0.200000003);
insert into order_details values (10836, 22, 21, 52, 0);
insert into order_details values (10836, 35, 18, 6, 0);
insert into order_details values (10836, 57, 19.5, 24, 0);
insert into order_details values (10836, 60, 34, 60, 0);
insert into order_details values (10836, 64, 33.25, 30, 0);
insert into order_details values (10837, 13, 6, 6, 0);
insert into order_details values (10837, 40, 18.3999996, 25, 0);
insert into order_details values (10837, 47, 9.5, 40, 0.25);
insert into order_details values (10837, 76, 18, 21, 0.25);
insert into order_details values (10838, 1, 18, 4, 0.25);
insert into order_details values (10838, 18, 62.5, 25, 0.25);
insert into order_details values (10838, 36, 19, 50, 0.25);
insert into order_details values (10839, 58, 13.25, 30, 0.100000001);
insert into order_details values (10839, 72, 34.7999992, 15, 0.100000001);
insert into order_details values (10840, 25, 14, 6, 0.200000003);
insert into order_details values (10840, 39, 18, 10, 0.200000003);
insert into order_details values (10841, 10, 31, 16, 0);
insert into order_details values (10841, 56, 38, 30, 0);
insert into order_details values (10841, 59, 55, 50, 0);
insert into order_details values (10841, 77, 13, 15, 0);
insert into order_details values (10842, 11, 21, 15, 0);
insert into order_details values (10842, 43, 46, 5, 0);
insert into order_details values (10842, 68, 12.5, 20, 0);
insert into order_details values (10842, 70, 15, 12, 0);
insert into order_details values (10843, 51, 53, 4, 0.25);
insert into order_details values (10844, 22, 21, 35, 0);
insert into order_details values (10845, 23, 9, 70, 0.100000001);
insert into order_details values (10845, 35, 18, 25, 0.100000001);
insert into order_details values (10845, 42, 14, 42, 0.100000001);
insert into order_details values (10845, 58, 13.25, 60, 0.100000001);
insert into order_details values (10845, 64, 33.25, 48, 0);
insert into order_details values (10846, 4, 22, 21, 0);
insert into order_details values (10846, 70, 15, 30, 0);
insert into order_details values (10846, 74, 10, 20, 0);
insert into order_details values (10847, 1, 18, 80, 0.200000003);
insert into order_details values (10847, 19, 9.19999981, 12, 0.200000003);
insert into order_details values (10847, 37, 26, 60, 0.200000003);
insert into order_details values (10847, 45, 9.5, 36, 0.200000003);
insert into order_details values (10847, 60, 34, 45, 0.200000003);
insert into order_details values (10847, 71, 21.5, 55, 0.200000003);
insert into order_details values (10848, 5, 21.3500004, 30, 0);
insert into order_details values (10848, 9, 97, 3, 0);
insert into order_details values (10849, 3, 10, 49, 0);
insert into order_details values (10849, 26, 31.2299995, 18, 0.150000006);
insert into order_details values (10850, 25, 14, 20, 0.150000006);
insert into order_details values (10850, 33, 2.5, 4, 0.150000006);
insert into order_details values (10850, 70, 15, 30, 0.150000006);
insert into order_details values (10851, 2, 19, 5, 0.0500000007);
insert into order_details values (10851, 25, 14, 10, 0.0500000007);
insert into order_details values (10851, 57, 19.5, 10, 0.0500000007);
insert into order_details values (10851, 59, 55, 42, 0.0500000007);
insert into order_details values (10852, 2, 19, 15, 0);
insert into order_details values (10852, 17, 39, 6, 0);
insert into order_details values (10852, 62, 49.2999992, 50, 0);
insert into order_details values (10853, 18, 62.5, 10, 0);
insert into order_details values (10854, 10, 31, 100, 0.150000006);
insert into order_details values (10854, 13, 6, 65, 0.150000006);
insert into order_details values (10855, 16, 17.4500008, 50, 0);
insert into order_details values (10855, 31, 12.5, 14, 0);
insert into order_details values (10855, 56, 38, 24, 0);
insert into order_details values (10855, 65, 21.0499992, 15, 0.150000006);
insert into order_details values (10856, 2, 19, 20, 0);
insert into order_details values (10856, 42, 14, 20, 0);
insert into order_details values (10857, 3, 10, 30, 0);
insert into order_details values (10857, 26, 31.2299995, 35, 0.25);
insert into order_details values (10857, 29, 123.790001, 10, 0.25);
insert into order_details values (10858, 7, 30, 5, 0);
insert into order_details values (10858, 27, 43.9000015, 10, 0);
insert into order_details values (10858, 70, 15, 4, 0);
insert into order_details values (10859, 24, 4.5, 40, 0.25);
insert into order_details values (10859, 54, 7.44999981, 35, 0.25);
insert into order_details values (10859, 64, 33.25, 30, 0.25);
insert into order_details values (10860, 51, 53, 3, 0);
insert into order_details values (10860, 76, 18, 20, 0);
insert into order_details values (10861, 17, 39, 42, 0);
insert into order_details values (10861, 18, 62.5, 20, 0);
insert into order_details values (10861, 21, 10, 40, 0);
insert into order_details values (10861, 33, 2.5, 35, 0);
insert into order_details values (10861, 62, 49.2999992, 3, 0);
insert into order_details values (10862, 11, 21, 25, 0);
insert into order_details values (10862, 52, 7, 8, 0);
insert into order_details values (10863, 1, 18, 20, 0.150000006);
insert into order_details values (10863, 58, 13.25, 12, 0.150000006);
insert into order_details values (10864, 35, 18, 4, 0);
insert into order_details values (10864, 67, 14, 15, 0);
insert into order_details values (10865, 38, 263.5, 60, 0.0500000007);
insert into order_details values (10865, 39, 18, 80, 0.0500000007);
insert into order_details values (10866, 2, 19, 21, 0.25);
insert into order_details values (10866, 24, 4.5, 6, 0.25);
insert into order_details values (10866, 30, 25.8899994, 40, 0.25);
insert into order_details values (10867, 53, 32.7999992, 3, 0);
insert into order_details values (10868, 26, 31.2299995, 20, 0);
insert into order_details values (10868, 35, 18, 30, 0);
insert into order_details values (10868, 49, 20, 42, 0.100000001);
insert into order_details values (10869, 1, 18, 40, 0);
insert into order_details values (10869, 11, 21, 10, 0);
insert into order_details values (10869, 23, 9, 50, 0);
insert into order_details values (10869, 68, 12.5, 20, 0);
insert into order_details values (10870, 35, 18, 3, 0);
insert into order_details values (10870, 51, 53, 2, 0);
insert into order_details values (10871, 6, 25, 50, 0.0500000007);
insert into order_details values (10871, 16, 17.4500008, 12, 0.0500000007);
insert into order_details values (10871, 17, 39, 16, 0.0500000007);
insert into order_details values (10872, 55, 24, 10, 0.0500000007);
insert into order_details values (10872, 62, 49.2999992, 20, 0.0500000007);
insert into order_details values (10872, 64, 33.25, 15, 0.0500000007);
insert into order_details values (10872, 65, 21.0499992, 21, 0.0500000007);
insert into order_details values (10873, 21, 10, 20, 0);
insert into order_details values (10873, 28, 45.5999985, 3, 0);
insert into order_details values (10874, 10, 31, 10, 0);
insert into order_details values (10875, 19, 9.19999981, 25, 0);
insert into order_details values (10875, 47, 9.5, 21, 0.100000001);
insert into order_details values (10875, 49, 20, 15, 0);
insert into order_details values (10876, 46, 12, 21, 0);
insert into order_details values (10876, 64, 33.25, 20, 0);
insert into order_details values (10877, 16, 17.4500008, 30, 0.25);
insert into order_details values (10877, 18, 62.5, 25, 0);
insert into order_details values (10878, 20, 81, 20, 0.0500000007);
insert into order_details values (10879, 40, 18.3999996, 12, 0);
insert into order_details values (10879, 65, 21.0499992, 10, 0);
insert into order_details values (10879, 76, 18, 10, 0);
insert into order_details values (10880, 23, 9, 30, 0.200000003);
insert into order_details values (10880, 61, 28.5, 30, 0.200000003);
insert into order_details values (10880, 70, 15, 50, 0.200000003);
insert into order_details values (10881, 73, 15, 10, 0);
insert into order_details values (10882, 42, 14, 25, 0);
insert into order_details values (10882, 49, 20, 20, 0.150000006);
insert into order_details values (10882, 54, 7.44999981, 32, 0.150000006);
insert into order_details values (10883, 24, 4.5, 8, 0);
insert into order_details values (10884, 21, 10, 40, 0.0500000007);
insert into order_details values (10884, 56, 38, 21, 0.0500000007);
insert into order_details values (10884, 65, 21.0499992, 12, 0.0500000007);
insert into order_details values (10885, 2, 19, 20, 0);
insert into order_details values (10885, 24, 4.5, 12, 0);
insert into order_details values (10885, 70, 15, 30, 0);
insert into order_details values (10885, 77, 13, 25, 0);
insert into order_details values (10886, 10, 31, 70, 0);
insert into order_details values (10886, 31, 12.5, 35, 0);
insert into order_details values (10886, 77, 13, 40, 0);
insert into order_details values (10887, 25, 14, 5, 0);
insert into order_details values (10888, 2, 19, 20, 0);
insert into order_details values (10888, 68, 12.5, 18, 0);
insert into order_details values (10889, 11, 21, 40, 0);
insert into order_details values (10889, 38, 263.5, 40, 0);
insert into order_details values (10890, 17, 39, 15, 0);
insert into order_details values (10890, 34, 14, 10, 0);
insert into order_details values (10890, 41, 9.64999962, 14, 0);
insert into order_details values (10891, 30, 25.8899994, 15, 0.0500000007);
insert into order_details values (10892, 59, 55, 40, 0.0500000007);
insert into order_details values (10893, 8, 40, 30, 0);
insert into order_details values (10893, 24, 4.5, 10, 0);
insert into order_details values (10893, 29, 123.790001, 24, 0);
insert into order_details values (10893, 30, 25.8899994, 35, 0);
insert into order_details values (10893, 36, 19, 20, 0);
insert into order_details values (10894, 13, 6, 28, 0.0500000007);
insert into order_details values (10894, 69, 36, 50, 0.0500000007);
insert into order_details values (10894, 75, 7.75, 120, 0.0500000007);
insert into order_details values (10895, 24, 4.5, 110, 0);
insert into order_details values (10895, 39, 18, 45, 0);
insert into order_details values (10895, 40, 18.3999996, 91, 0);
insert into order_details values (10895, 60, 34, 100, 0);
insert into order_details values (10896, 45, 9.5, 15, 0);
insert into order_details values (10896, 56, 38, 16, 0);
insert into order_details values (10897, 29, 123.790001, 80, 0);
insert into order_details values (10897, 30, 25.8899994, 36, 0);
insert into order_details values (10898, 13, 6, 5, 0);
insert into order_details values (10899, 39, 18, 8, 0.150000006);
insert into order_details values (10900, 70, 15, 3, 0.25);
insert into order_details values (10901, 41, 9.64999962, 30, 0);
insert into order_details values (10901, 71, 21.5, 30, 0);
insert into order_details values (10902, 55, 24, 30, 0.150000006);
insert into order_details values (10902, 62, 49.2999992, 6, 0.150000006);
insert into order_details values (10903, 13, 6, 40, 0);
insert into order_details values (10903, 65, 21.0499992, 21, 0);
insert into order_details values (10903, 68, 12.5, 20, 0);
insert into order_details values (10904, 58, 13.25, 15, 0);
insert into order_details values (10904, 62, 49.2999992, 35, 0);
insert into order_details values (10905, 1, 18, 20, 0.0500000007);
insert into order_details values (10906, 61, 28.5, 15, 0);
insert into order_details values (10907, 75, 7.75, 14, 0);
insert into order_details values (10908, 7, 30, 20, 0.0500000007);
insert into order_details values (10908, 52, 7, 14, 0.0500000007);
insert into order_details values (10909, 7, 30, 12, 0);
insert into order_details values (10909, 16, 17.4500008, 15, 0);
insert into order_details values (10909, 41, 9.64999962, 5, 0);
insert into order_details values (10910, 19, 9.19999981, 12, 0);
insert into order_details values (10910, 49, 20, 10, 0);
insert into order_details values (10910, 61, 28.5, 5, 0);
insert into order_details values (10911, 1, 18, 10, 0);
insert into order_details values (10911, 17, 39, 12, 0);
insert into order_details values (10911, 67, 14, 15, 0);
insert into order_details values (10912, 11, 21, 40, 0.25);
insert into order_details values (10912, 29, 123.790001, 60, 0.25);
insert into order_details values (10913, 4, 22, 30, 0.25);
insert into order_details values (10913, 33, 2.5, 40, 0.25);
insert into order_details values (10913, 58, 13.25, 15, 0);
insert into order_details values (10914, 71, 21.5, 25, 0);
insert into order_details values (10915, 17, 39, 10, 0);
insert into order_details values (10915, 33, 2.5, 30, 0);
insert into order_details values (10915, 54, 7.44999981, 10, 0);
insert into order_details values (10916, 16, 17.4500008, 6, 0);
insert into order_details values (10916, 32, 32, 6, 0);
insert into order_details values (10916, 57, 19.5, 20, 0);
insert into order_details values (10917, 30, 25.8899994, 1, 0);
insert into order_details values (10917, 60, 34, 10, 0);
insert into order_details values (10918, 1, 18, 60, 0.25);
insert into order_details values (10918, 60, 34, 25, 0.25);
insert into order_details values (10919, 16, 17.4500008, 24, 0);
insert into order_details values (10919, 25, 14, 24, 0);
insert into order_details values (10919, 40, 18.3999996, 20, 0);
insert into order_details values (10920, 50, 16.25, 24, 0);
insert into order_details values (10921, 35, 18, 10, 0);
insert into order_details values (10921, 63, 43.9000015, 40, 0);
insert into order_details values (10922, 17, 39, 15, 0);
insert into order_details values (10922, 24, 4.5, 35, 0);
insert into order_details values (10923, 42, 14, 10, 0.200000003);
insert into order_details values (10923, 43, 46, 10, 0.200000003);
insert into order_details values (10923, 67, 14, 24, 0.200000003);
insert into order_details values (10924, 10, 31, 20, 0.100000001);
insert into order_details values (10924, 28, 45.5999985, 30, 0.100000001);
insert into order_details values (10924, 75, 7.75, 6, 0);
insert into order_details values (10925, 36, 19, 25, 0.150000006);
insert into order_details values (10925, 52, 7, 12, 0.150000006);
insert into order_details values (10926, 11, 21, 2, 0);
insert into order_details values (10926, 13, 6, 10, 0);
insert into order_details values (10926, 19, 9.19999981, 7, 0);
insert into order_details values (10926, 72, 34.7999992, 10, 0);
insert into order_details values (10927, 20, 81, 5, 0);
insert into order_details values (10927, 52, 7, 5, 0);
insert into order_details values (10927, 76, 18, 20, 0);
insert into order_details values (10928, 47, 9.5, 5, 0);
insert into order_details values (10928, 76, 18, 5, 0);
insert into order_details values (10929, 21, 10, 60, 0);
insert into order_details values (10929, 75, 7.75, 49, 0);
insert into order_details values (10929, 77, 13, 15, 0);
insert into order_details values (10930, 21, 10, 36, 0);
insert into order_details values (10930, 27, 43.9000015, 25, 0);
insert into order_details values (10930, 55, 24, 25, 0.200000003);
insert into order_details values (10930, 58, 13.25, 30, 0.200000003);
insert into order_details values (10931, 13, 6, 42, 0.150000006);
insert into order_details values (10931, 57, 19.5, 30, 0);
insert into order_details values (10932, 16, 17.4500008, 30, 0.100000001);
insert into order_details values (10932, 62, 49.2999992, 14, 0.100000001);
insert into order_details values (10932, 72, 34.7999992, 16, 0);
insert into order_details values (10932, 75, 7.75, 20, 0.100000001);
insert into order_details values (10933, 53, 32.7999992, 2, 0);
insert into order_details values (10933, 61, 28.5, 30, 0);
insert into order_details values (10934, 6, 25, 20, 0);
insert into order_details values (10935, 1, 18, 21, 0);
insert into order_details values (10935, 18, 62.5, 4, 0.25);
insert into order_details values (10935, 23, 9, 8, 0.25);
insert into order_details values (10936, 36, 19, 30, 0.200000003);
insert into order_details values (10937, 28, 45.5999985, 8, 0);
insert into order_details values (10937, 34, 14, 20, 0);
insert into order_details values (10938, 13, 6, 20, 0.25);
insert into order_details values (10938, 43, 46, 24, 0.25);
insert into order_details values (10938, 60, 34, 49, 0.25);
insert into order_details values (10938, 71, 21.5, 35, 0.25);
insert into order_details values (10939, 2, 19, 10, 0.150000006);
insert into order_details values (10939, 67, 14, 40, 0.150000006);
insert into order_details values (10940, 7, 30, 8, 0);
insert into order_details values (10940, 13, 6, 20, 0);
insert into order_details values (10941, 31, 12.5, 44, 0.25);
insert into order_details values (10941, 62, 49.2999992, 30, 0.25);
insert into order_details values (10941, 68, 12.5, 80, 0.25);
insert into order_details values (10941, 72, 34.7999992, 50, 0);
insert into order_details values (10942, 49, 20, 28, 0);
insert into order_details values (10943, 13, 6, 15, 0);
insert into order_details values (10943, 22, 21, 21, 0);
insert into order_details values (10943, 46, 12, 15, 0);
insert into order_details values (10944, 11, 21, 5, 0.25);
insert into order_details values (10944, 44, 19.4500008, 18, 0.25);
insert into order_details values (10944, 56, 38, 18, 0);
insert into order_details values (10945, 13, 6, 20, 0);
insert into order_details values (10945, 31, 12.5, 10, 0);
insert into order_details values (10946, 10, 31, 25, 0);
insert into order_details values (10946, 24, 4.5, 25, 0);
insert into order_details values (10946, 77, 13, 40, 0);
insert into order_details values (10947, 59, 55, 4, 0);
insert into order_details values (10948, 50, 16.25, 9, 0);
insert into order_details values (10948, 51, 53, 40, 0);
insert into order_details values (10948, 55, 24, 4, 0);
insert into order_details values (10949, 6, 25, 12, 0);
insert into order_details values (10949, 10, 31, 30, 0);
insert into order_details values (10949, 17, 39, 6, 0);
insert into order_details values (10949, 62, 49.2999992, 60, 0);
insert into order_details values (10950, 4, 22, 5, 0);
insert into order_details values (10951, 33, 2.5, 15, 0.0500000007);
insert into order_details values (10951, 41, 9.64999962, 6, 0.0500000007);
insert into order_details values (10951, 75, 7.75, 50, 0.0500000007);
insert into order_details values (10952, 6, 25, 16, 0.0500000007);
insert into order_details values (10952, 28, 45.5999985, 2, 0);
insert into order_details values (10953, 20, 81, 50, 0.0500000007);
insert into order_details values (10953, 31, 12.5, 50, 0.0500000007);
insert into order_details values (10954, 16, 17.4500008, 28, 0.150000006);
insert into order_details values (10954, 31, 12.5, 25, 0.150000006);
insert into order_details values (10954, 45, 9.5, 30, 0);
insert into order_details values (10954, 60, 34, 24, 0.150000006);
insert into order_details values (10955, 75, 7.75, 12, 0.200000003);
insert into order_details values (10956, 21, 10, 12, 0);
insert into order_details values (10956, 47, 9.5, 14, 0);
insert into order_details values (10956, 51, 53, 8, 0);
insert into order_details values (10957, 30, 25.8899994, 30, 0);
insert into order_details values (10957, 35, 18, 40, 0);
insert into order_details values (10957, 64, 33.25, 8, 0);
insert into order_details values (10958, 5, 21.3500004, 20, 0);
insert into order_details values (10958, 7, 30, 6, 0);
insert into order_details values (10958, 72, 34.7999992, 5, 0);
insert into order_details values (10959, 75, 7.75, 20, 0.150000006);
insert into order_details values (10960, 24, 4.5, 10, 0.25);
insert into order_details values (10960, 41, 9.64999962, 24, 0);
insert into order_details values (10961, 52, 7, 6, 0.0500000007);
insert into order_details values (10961, 76, 18, 60, 0);
insert into order_details values (10962, 7, 30, 45, 0);
insert into order_details values (10962, 13, 6, 77, 0);
insert into order_details values (10962, 53, 32.7999992, 20, 0);
insert into order_details values (10962, 69, 36, 9, 0);
insert into order_details values (10962, 76, 18, 44, 0);
insert into order_details values (10963, 60, 34, 2, 0.150000006);
insert into order_details values (10964, 18, 62.5, 6, 0);
insert into order_details values (10964, 38, 263.5, 5, 0);
insert into order_details values (10964, 69, 36, 10, 0);
insert into order_details values (10965, 51, 53, 16, 0);
insert into order_details values (10966, 37, 26, 8, 0);
insert into order_details values (10966, 56, 38, 12, 0.150000006);
insert into order_details values (10966, 62, 49.2999992, 12, 0.150000006);
insert into order_details values (10967, 19, 9.19999981, 12, 0);
insert into order_details values (10967, 49, 20, 40, 0);
insert into order_details values (10968, 12, 38, 30, 0);
insert into order_details values (10968, 24, 4.5, 30, 0);
insert into order_details values (10968, 64, 33.25, 4, 0);
insert into order_details values (10969, 46, 12, 9, 0);
insert into order_details values (10970, 52, 7, 40, 0.200000003);
insert into order_details values (10971, 29, 123.790001, 14, 0);
insert into order_details values (10972, 17, 39, 6, 0);
insert into order_details values (10972, 33, 2.5, 7, 0);
insert into order_details values (10973, 26, 31.2299995, 5, 0);
insert into order_details values (10973, 41, 9.64999962, 6, 0);
insert into order_details values (10973, 75, 7.75, 10, 0);
insert into order_details values (10974, 63, 43.9000015, 10, 0);
insert into order_details values (10975, 8, 40, 16, 0);
insert into order_details values (10975, 75, 7.75, 10, 0);
insert into order_details values (10976, 28, 45.5999985, 20, 0);
insert into order_details values (10977, 39, 18, 30, 0);
insert into order_details values (10977, 47, 9.5, 30, 0);
insert into order_details values (10977, 51, 53, 10, 0);
insert into order_details values (10977, 63, 43.9000015, 20, 0);
insert into order_details values (10978, 8, 40, 20, 0.150000006);
insert into order_details values (10978, 21, 10, 40, 0.150000006);
insert into order_details values (10978, 40, 18.3999996, 10, 0);
insert into order_details values (10978, 44, 19.4500008, 6, 0.150000006);
insert into order_details values (10979, 7, 30, 18, 0);
insert into order_details values (10979, 12, 38, 20, 0);
insert into order_details values (10979, 24, 4.5, 80, 0);
insert into order_details values (10979, 27, 43.9000015, 30, 0);
insert into order_details values (10979, 31, 12.5, 24, 0);
insert into order_details values (10979, 63, 43.9000015, 35, 0);
insert into order_details values (10980, 75, 7.75, 40, 0.200000003);
insert into order_details values (10981, 38, 263.5, 60, 0);
insert into order_details values (10982, 7, 30, 20, 0);
insert into order_details values (10982, 43, 46, 9, 0);
insert into order_details values (10983, 13, 6, 84, 0.150000006);
insert into order_details values (10983, 57, 19.5, 15, 0);
insert into order_details values (10984, 16, 17.4500008, 55, 0);
insert into order_details values (10984, 24, 4.5, 20, 0);
insert into order_details values (10984, 36, 19, 40, 0);
insert into order_details values (10985, 16, 17.4500008, 36, 0.100000001);
insert into order_details values (10985, 18, 62.5, 8, 0.100000001);
insert into order_details values (10985, 32, 32, 35, 0.100000001);
insert into order_details values (10986, 11, 21, 30, 0);
insert into order_details values (10986, 20, 81, 15, 0);
insert into order_details values (10986, 76, 18, 10, 0);
insert into order_details values (10986, 77, 13, 15, 0);
insert into order_details values (10987, 7, 30, 60, 0);
insert into order_details values (10987, 43, 46, 6, 0);
insert into order_details values (10987, 72, 34.7999992, 20, 0);
insert into order_details values (10988, 7, 30, 60, 0);
insert into order_details values (10988, 62, 49.2999992, 40, 0.100000001);
insert into order_details values (10989, 6, 25, 40, 0);
insert into order_details values (10989, 11, 21, 15, 0);
insert into order_details values (10989, 41, 9.64999962, 4, 0);
insert into order_details values (10990, 21, 10, 65, 0);
insert into order_details values (10990, 34, 14, 60, 0.150000006);
insert into order_details values (10990, 55, 24, 65, 0.150000006);
insert into order_details values (10990, 61, 28.5, 66, 0.150000006);
insert into order_details values (10991, 2, 19, 50, 0.200000003);
insert into order_details values (10991, 70, 15, 20, 0.200000003);
insert into order_details values (10991, 76, 18, 90, 0.200000003);
insert into order_details values (10992, 72, 34.7999992, 2, 0);
insert into order_details values (10993, 29, 123.790001, 50, 0.25);
insert into order_details values (10993, 41, 9.64999962, 35, 0.25);
insert into order_details values (10994, 59, 55, 18, 0.0500000007);
insert into order_details values (10995, 51, 53, 20, 0);
insert into order_details values (10995, 60, 34, 4, 0);
insert into order_details values (10996, 42, 14, 40, 0);
insert into order_details values (10997, 32, 32, 50, 0);
insert into order_details values (10997, 46, 12, 20, 0.25);
insert into order_details values (10997, 52, 7, 20, 0.25);
insert into order_details values (10998, 24, 4.5, 12, 0);
insert into order_details values (10998, 61, 28.5, 7, 0);
insert into order_details values (10998, 74, 10, 20, 0);
insert into order_details values (10998, 75, 7.75, 30, 0);
insert into order_details values (10999, 41, 9.64999962, 20, 0.0500000007);
insert into order_details values (10999, 51, 53, 15, 0.0500000007);
insert into order_details values (10999, 77, 13, 21, 0.0500000007);
insert into order_details values (11000, 4, 22, 25, 0.25);
insert into order_details values (11000, 24, 4.5, 30, 0.25);
insert into order_details values (11000, 77, 13, 30, 0);
insert into order_details values (11001, 7, 30, 60, 0);
insert into order_details values (11001, 22, 21, 25, 0);
insert into order_details values (11001, 46, 12, 25, 0);
insert into order_details values (11001, 55, 24, 6, 0);
insert into order_details values (11002, 13, 6, 56, 0);
insert into order_details values (11002, 35, 18, 15, 0.150000006);
insert into order_details values (11002, 42, 14, 24, 0.150000006);
insert into order_details values (11002, 55, 24, 40, 0);
insert into order_details values (11003, 1, 18, 4, 0);
insert into order_details values (11003, 40, 18.3999996, 10, 0);
insert into order_details values (11003, 52, 7, 10, 0);
insert into order_details values (11004, 26, 31.2299995, 6, 0);
insert into order_details values (11004, 76, 18, 6, 0);
insert into order_details values (11005, 1, 18, 2, 0);
insert into order_details values (11005, 59, 55, 10, 0);
insert into order_details values (11006, 1, 18, 8, 0);
insert into order_details values (11006, 29, 123.790001, 2, 0.25);
insert into order_details values (11007, 8, 40, 30, 0);
insert into order_details values (11007, 29, 123.790001, 10, 0);
insert into order_details values (11007, 42, 14, 14, 0);
insert into order_details values (11008, 28, 45.5999985, 70, 0.0500000007);
insert into order_details values (11008, 34, 14, 90, 0.0500000007);
insert into order_details values (11008, 71, 21.5, 21, 0);
insert into order_details values (11009, 24, 4.5, 12, 0);
insert into order_details values (11009, 36, 19, 18, 0.25);
insert into order_details values (11009, 60, 34, 9, 0);
insert into order_details values (11010, 7, 30, 20, 0);
insert into order_details values (11010, 24, 4.5, 10, 0);
insert into order_details values (11011, 58, 13.25, 40, 0.0500000007);
insert into order_details values (11011, 71, 21.5, 20, 0);
insert into order_details values (11012, 19, 9.19999981, 50, 0.0500000007);
insert into order_details values (11012, 60, 34, 36, 0.0500000007);
insert into order_details values (11012, 71, 21.5, 60, 0.0500000007);
insert into order_details values (11013, 23, 9, 10, 0);
insert into order_details values (11013, 42, 14, 4, 0);
insert into order_details values (11013, 45, 9.5, 20, 0);
insert into order_details values (11013, 68, 12.5, 2, 0);
insert into order_details values (11014, 41, 9.64999962, 28, 0.100000001);
insert into order_details values (11015, 30, 25.8899994, 15, 0);
insert into order_details values (11015, 77, 13, 18, 0);
insert into order_details values (11016, 31, 12.5, 15, 0);
insert into order_details values (11016, 36, 19, 16, 0);
insert into order_details values (11017, 3, 10, 25, 0);
insert into order_details values (11017, 59, 55, 110, 0);
insert into order_details values (11017, 70, 15, 30, 0);
insert into order_details values (11018, 12, 38, 20, 0);
insert into order_details values (11018, 18, 62.5, 10, 0);
insert into order_details values (11018, 56, 38, 5, 0);
insert into order_details values (11019, 46, 12, 3, 0);
insert into order_details values (11019, 49, 20, 2, 0);
insert into order_details values (11020, 10, 31, 24, 0.150000006);
insert into order_details values (11021, 2, 19, 11, 0.25);
insert into order_details values (11021, 20, 81, 15, 0);
insert into order_details values (11021, 26, 31.2299995, 63, 0);
insert into order_details values (11021, 51, 53, 44, 0.25);
insert into order_details values (11021, 72, 34.7999992, 35, 0);
insert into order_details values (11022, 19, 9.19999981, 35, 0);
insert into order_details values (11022, 69, 36, 30, 0);
insert into order_details values (11023, 7, 30, 4, 0);
insert into order_details values (11023, 43, 46, 30, 0);
insert into order_details values (11024, 26, 31.2299995, 12, 0);
insert into order_details values (11024, 33, 2.5, 30, 0);
insert into order_details values (11024, 65, 21.0499992, 21, 0);
insert into order_details values (11024, 71, 21.5, 50, 0);
insert into order_details values (11025, 1, 18, 10, 0.100000001);
insert into order_details values (11025, 13, 6, 20, 0.100000001);
insert into order_details values (11026, 18, 62.5, 8, 0);
insert into order_details values (11026, 51, 53, 10, 0);
insert into order_details values (11027, 24, 4.5, 30, 0.25);
insert into order_details values (11027, 62, 49.2999992, 21, 0.25);
insert into order_details values (11028, 55, 24, 35, 0);
insert into order_details values (11028, 59, 55, 24, 0);
insert into order_details values (11029, 56, 38, 20, 0);
insert into order_details values (11029, 63, 43.9000015, 12, 0);
insert into order_details values (11030, 2, 19, 100, 0.25);
insert into order_details values (11030, 5, 21.3500004, 70, 0);
insert into order_details values (11030, 29, 123.790001, 60, 0.25);
insert into order_details values (11030, 59, 55, 100, 0.25);
insert into order_details values (11031, 1, 18, 45, 0);
insert into order_details values (11031, 13, 6, 80, 0);
insert into order_details values (11031, 24, 4.5, 21, 0);
insert into order_details values (11031, 64, 33.25, 20, 0);
insert into order_details values (11031, 71, 21.5, 16, 0);
insert into order_details values (11032, 36, 19, 35, 0);
insert into order_details values (11032, 38, 263.5, 25, 0);
insert into order_details values (11032, 59, 55, 30, 0);
insert into order_details values (11033, 53, 32.7999992, 70, 0.100000001);
insert into order_details values (11033, 69, 36, 36, 0.100000001);
insert into order_details values (11034, 21, 10, 15, 0.100000001);
insert into order_details values (11034, 44, 19.4500008, 12, 0);
insert into order_details values (11034, 61, 28.5, 6, 0);
insert into order_details values (11035, 1, 18, 10, 0);
insert into order_details values (11035, 35, 18, 60, 0);
insert into order_details values (11035, 42, 14, 30, 0);
insert into order_details values (11035, 54, 7.44999981, 10, 0);
insert into order_details values (11036, 13, 6, 7, 0);
insert into order_details values (11036, 59, 55, 30, 0);
insert into order_details values (11037, 70, 15, 4, 0);
insert into order_details values (11038, 40, 18.3999996, 5, 0.200000003);
insert into order_details values (11038, 52, 7, 2, 0);
insert into order_details values (11038, 71, 21.5, 30, 0);
insert into order_details values (11039, 28, 45.5999985, 20, 0);
insert into order_details values (11039, 35, 18, 24, 0);
insert into order_details values (11039, 49, 20, 60, 0);
insert into order_details values (11039, 57, 19.5, 28, 0);
insert into order_details values (11040, 21, 10, 20, 0);
insert into order_details values (11041, 2, 19, 30, 0.200000003);
insert into order_details values (11041, 63, 43.9000015, 30, 0);
insert into order_details values (11042, 44, 19.4500008, 15, 0);
insert into order_details values (11042, 61, 28.5, 4, 0);
insert into order_details values (11043, 11, 21, 10, 0);
insert into order_details values (11044, 62, 49.2999992, 12, 0);
insert into order_details values (11045, 33, 2.5, 15, 0);
insert into order_details values (11045, 51, 53, 24, 0);
insert into order_details values (11046, 12, 38, 20, 0.0500000007);
insert into order_details values (11046, 32, 32, 15, 0.0500000007);
insert into order_details values (11046, 35, 18, 18, 0.0500000007);
insert into order_details values (11047, 1, 18, 25, 0.25);
insert into order_details values (11047, 5, 21.3500004, 30, 0.25);
insert into order_details values (11048, 68, 12.5, 42, 0);
insert into order_details values (11049, 2, 19, 10, 0.200000003);
insert into order_details values (11049, 12, 38, 4, 0.200000003);
insert into order_details values (11050, 76, 18, 50, 0.100000001);
insert into order_details values (11051, 24, 4.5, 10, 0.200000003);
insert into order_details values (11052, 43, 46, 30, 0.200000003);
insert into order_details values (11052, 61, 28.5, 10, 0.200000003);
insert into order_details values (11053, 18, 62.5, 35, 0.200000003);
insert into order_details values (11053, 32, 32, 20, 0);
insert into order_details values (11053, 64, 33.25, 25, 0.200000003);
insert into order_details values (11054, 33, 2.5, 10, 0);
insert into order_details values (11054, 67, 14, 20, 0);
insert into order_details values (11055, 24, 4.5, 15, 0);
insert into order_details values (11055, 25, 14, 15, 0);
insert into order_details values (11055, 51, 53, 20, 0);
insert into order_details values (11055, 57, 19.5, 20, 0);
insert into order_details values (11056, 7, 30, 40, 0);
insert into order_details values (11056, 55, 24, 35, 0);
insert into order_details values (11056, 60, 34, 50, 0);
insert into order_details values (11057, 70, 15, 3, 0);
insert into order_details values (11058, 21, 10, 3, 0);
insert into order_details values (11058, 60, 34, 21, 0);
insert into order_details values (11058, 61, 28.5, 4, 0);
insert into order_details values (11059, 13, 6, 30, 0);
insert into order_details values (11059, 17, 39, 12, 0);
insert into order_details values (11059, 60, 34, 35, 0);
insert into order_details values (11060, 60, 34, 4, 0);
insert into order_details values (11060, 77, 13, 10, 0);
insert into order_details values (11061, 60, 34, 15, 0);
insert into order_details values (11062, 53, 32.7999992, 10, 0.200000003);
insert into order_details values (11062, 70, 15, 12, 0.200000003);
insert into order_details values (11063, 34, 14, 30, 0);
insert into order_details values (11063, 40, 18.3999996, 40, 0.100000001);
insert into order_details values (11063, 41, 9.64999962, 30, 0.100000001);
insert into order_details values (11064, 17, 39, 77, 0.100000001);
insert into order_details values (11064, 41, 9.64999962, 12, 0);
insert into order_details values (11064, 53, 32.7999992, 25, 0.100000001);
insert into order_details values (11064, 55, 24, 4, 0.100000001);
insert into order_details values (11064, 68, 12.5, 55, 0);
insert into order_details values (11065, 30, 25.8899994, 4, 0.25);
insert into order_details values (11065, 54, 7.44999981, 20, 0.25);
insert into order_details values (11066, 16, 17.4500008, 3, 0);
insert into order_details values (11066, 19, 9.19999981, 42, 0);
insert into order_details values (11066, 34, 14, 35, 0);
insert into order_details values (11067, 41, 9.64999962, 9, 0);
insert into order_details values (11068, 28, 45.5999985, 8, 0.150000006);
insert into order_details values (11068, 43, 46, 36, 0.150000006);
insert into order_details values (11068, 77, 13, 28, 0.150000006);
insert into order_details values (11069, 39, 18, 20, 0);
insert into order_details values (11070, 1, 18, 40, 0.150000006);
insert into order_details values (11070, 2, 19, 20, 0.150000006);
insert into order_details values (11070, 16, 17.4500008, 30, 0.150000006);
insert into order_details values (11070, 31, 12.5, 20, 0);
insert into order_details values (11071, 7, 30, 15, 0.0500000007);
insert into order_details values (11071, 13, 6, 10, 0.0500000007);
insert into order_details values (11072, 2, 19, 8, 0);
insert into order_details values (11072, 41, 9.64999962, 40, 0);
insert into order_details values (11072, 50, 16.25, 22, 0);
insert into order_details values (11072, 64, 33.25, 130, 0);
insert into order_details values (11073, 11, 21, 10, 0);
insert into order_details values (11073, 24, 4.5, 20, 0);
insert into order_details values (11074, 16, 17.4500008, 14, 0.0500000007);
insert into order_details values (11075, 2, 19, 10, 0.150000006);
insert into order_details values (11075, 46, 12, 30, 0.150000006);
insert into order_details values (11075, 76, 18, 2, 0.150000006);
insert into order_details values (11076, 6, 25, 20, 0.25);
insert into order_details values (11076, 14, 23.25, 20, 0.25);
insert into order_details values (11076, 19, 9.19999981, 10, 0.25);
insert into order_details values (11077, 2, 19, 24, 0.200000003);
insert into order_details values (11077, 3, 10, 4, 0);
insert into order_details values (11077, 4, 22, 1, 0);
insert into order_details values (11077, 6, 25, 1, 0.0199999996);
insert into order_details values (11077, 7, 30, 1, 0.0500000007);
insert into order_details values (11077, 8, 40, 2, 0.100000001);
insert into order_details values (11077, 10, 31, 1, 0);
insert into order_details values (11077, 12, 38, 2, 0.0500000007);
insert into order_details values (11077, 13, 6, 4, 0);
insert into order_details values (11077, 14, 23.25, 1, 0.0299999993);
insert into order_details values (11077, 16, 17.4500008, 2, 0.0299999993);
insert into order_details values (11077, 20, 81, 1, 0.0399999991);
insert into order_details values (11077, 23, 9, 2, 0);
insert into order_details values (11077, 32, 32, 1, 0);
insert into order_details values (11077, 39, 18, 2, 0.0500000007);
insert into order_details values (11077, 41, 9.64999962, 3, 0);
insert into order_details values (11077, 46, 12, 3, 0.0199999996);
insert into order_details values (11077, 52, 7, 2, 0);
insert into order_details values (11077, 55, 24, 2, 0);
insert into order_details values (11077, 60, 34, 2, 0.0599999987);
insert into order_details values (11077, 64, 33.25, 2, 0.0299999993);
insert into order_details values (11077, 66, 17, 1, 0);
insert into order_details values (11077, 73, 15, 2, 0.00999999978);
insert into order_details values (11077, 75, 7.75, 4, 0);
insert into order_details values (11077, 77, 13, 2, 0);


--
-- data for name: orders; type: table data; schema: public; owner: postgres
--

insert into orders values (10248, 'vinet', 5, '1996-07-04', '1996-08-01', '1996-07-16', 3, 32.3800011, 'vins et alcools chevalier', '59 rue de l''abbaye', 'reims', null, '51100', 'france');
insert into orders values (10249, 'tomsp', 6, '1996-07-05', '1996-08-16', '1996-07-10', 1, 11.6099997, 'toms spezialitäten', 'luisenstr. 48', 'münster', null, '44087', 'germany');
insert into orders values (10250, 'hanar', 4, '1996-07-08', '1996-08-05', '1996-07-12', 2, 65.8300018, 'hanari carnes', 'rua do paço, 67', 'rio de janeiro', 'rj', '05454-876', 'brazil');
insert into orders values (10251, 'victe', 3, '1996-07-08', '1996-08-05', '1996-07-15', 1, 41.3400002, 'victuailles en stock', '2, rue du commerce', 'lyon', null, '69004', 'france');
insert into orders values (10252, 'suprd', 4, '1996-07-09', '1996-08-06', '1996-07-11', 2, 51.2999992, 'suprêmes délices', 'boulevard tirou, 255', 'charleroi', null, 'b-6000', 'belgium');
insert into orders values (10253, 'hanar', 3, '1996-07-10', '1996-07-24', '1996-07-16', 2, 58.1699982, 'hanari carnes', 'rua do paço, 67', 'rio de janeiro', 'rj', '05454-876', 'brazil');
insert into orders values (10254, 'chops', 5, '1996-07-11', '1996-08-08', '1996-07-23', 2, 22.9799995, 'chop-suey chinese', 'hauptstr. 31', 'bern', null, '3012', 'switzerland');
insert into orders values (10255, 'ricsu', 9, '1996-07-12', '1996-08-09', '1996-07-15', 3, 148.330002, 'richter supermarkt', 'starenweg 5', 'genève', null, '1204', 'switzerland');
insert into orders values (10256, 'welli', 3, '1996-07-15', '1996-08-12', '1996-07-17', 2, 13.9700003, 'wellington importadora', 'rua do mercado, 12', 'resende', 'sp', '08737-363', 'brazil');
insert into orders values (10257, 'hilaa', 4, '1996-07-16', '1996-08-13', '1996-07-22', 3, 81.9100037, 'hilarion-abastos', 'carrera 22 con ave. carlos soublette #8-35', 'san cristóbal', 'táchira', '5022', 'venezuela');
insert into orders values (10258, 'ernsh', 1, '1996-07-17', '1996-08-14', '1996-07-23', 1, 140.509995, 'ernst handel', 'kirchgasse 6', 'graz', null, '8010', 'austria');
insert into orders values (10259, 'centc', 4, '1996-07-18', '1996-08-15', '1996-07-25', 3, 3.25, 'centro comercial moctezuma', 'sierras de granada 9993', 'méxico d.f.', null, '05022', 'mexico');
insert into orders values (10260, 'ottik', 4, '1996-07-19', '1996-08-16', '1996-07-29', 1, 55.0900002, 'ottilies käseladen', 'mehrheimerstr. 369', 'köln', null, '50739', 'germany');
insert into orders values (10261, 'quede', 4, '1996-07-19', '1996-08-16', '1996-07-30', 2, 3.04999995, 'que delícia', 'rua da panificadora, 12', 'rio de janeiro', 'rj', '02389-673', 'brazil');
insert into orders values (10262, 'rattc', 8, '1996-07-22', '1996-08-19', '1996-07-25', 3, 48.2900009, 'rattlesnake canyon grocery', '2817 milton dr.', 'albuquerque', 'nm', '87110', 'usa');
insert into orders values (10263, 'ernsh', 9, '1996-07-23', '1996-08-20', '1996-07-31', 3, 146.059998, 'ernst handel', 'kirchgasse 6', 'graz', null, '8010', 'austria');
insert into orders values (10264, 'folko', 6, '1996-07-24', '1996-08-21', '1996-08-23', 3, 3.67000008, 'folk och fä hb', 'åkergatan 24', 'bräcke', null, 's-844 67', 'sweden');
insert into orders values (10265, 'blonp', 2, '1996-07-25', '1996-08-22', '1996-08-12', 1, 55.2799988, 'blondel père et fils', '24, place kléber', 'strasbourg', null, '67000', 'france');
insert into orders values (10266, 'warth', 3, '1996-07-26', '1996-09-06', '1996-07-31', 3, 25.7299995, 'wartian herkku', 'torikatu 38', 'oulu', null, '90110', 'finland');
insert into orders values (10267, 'frank', 4, '1996-07-29', '1996-08-26', '1996-08-06', 1, 208.580002, 'frankenversand', 'berliner platz 43', 'münchen', null, '80805', 'germany');
insert into orders values (10268, 'grosr', 8, '1996-07-30', '1996-08-27', '1996-08-02', 3, 66.2900009, 'grosella-restaurante', '5ª ave. los palos grandes', 'caracas', 'df', '1081', 'venezuela');
insert into orders values (10269, 'whitc', 5, '1996-07-31', '1996-08-14', '1996-08-09', 1, 4.55999994, 'white clover markets', '1029 - 12th ave. s.', 'seattle', 'wa', '98124', 'usa');
insert into orders values (10270, 'warth', 1, '1996-08-01', '1996-08-29', '1996-08-02', 1, 136.539993, 'wartian herkku', 'torikatu 38', 'oulu', null, '90110', 'finland');
insert into orders values (10271, 'splir', 6, '1996-08-01', '1996-08-29', '1996-08-30', 2, 4.53999996, 'split rail beer & ale', 'p.o. box 555', 'lander', 'wy', '82520', 'usa');
insert into orders values (10272, 'rattc', 6, '1996-08-02', '1996-08-30', '1996-08-06', 2, 98.0299988, 'rattlesnake canyon grocery', '2817 milton dr.', 'albuquerque', 'nm', '87110', 'usa');
insert into orders values (10273, 'quick', 3, '1996-08-05', '1996-09-02', '1996-08-12', 3, 76.0699997, 'quick-stop', 'taucherstraße 10', 'cunewalde', null, '01307', 'germany');
insert into orders values (10274, 'vinet', 6, '1996-08-06', '1996-09-03', '1996-08-16', 1, 6.01000023, 'vins et alcools chevalier', '59 rue de l''abbaye', 'reims', null, '51100', 'france');
insert into orders values (10275, 'magaa', 1, '1996-08-07', '1996-09-04', '1996-08-09', 1, 26.9300003, 'magazzini alimentari riuniti', 'via ludovico il moro 22', 'bergamo', null, '24100', 'italy');
insert into orders values (10276, 'tortu', 8, '1996-08-08', '1996-08-22', '1996-08-14', 3, 13.8400002, 'tortuga restaurante', 'avda. azteca 123', 'méxico d.f.', null, '05033', 'mexico');
insert into orders values (10277, 'morgk', 2, '1996-08-09', '1996-09-06', '1996-08-13', 3, 125.769997, 'morgenstern gesundkost', 'heerstr. 22', 'leipzig', null, '04179', 'germany');
insert into orders values (10278, 'bergs', 8, '1996-08-12', '1996-09-09', '1996-08-16', 2, 92.6900024, 'berglunds snabbköp', 'berguvsvägen  8', 'luleå', null, 's-958 22', 'sweden');
insert into orders values (10279, 'lehms', 8, '1996-08-13', '1996-09-10', '1996-08-16', 2, 25.8299999, 'lehmanns marktstand', 'magazinweg 7', 'frankfurt a.m.', null, '60528', 'germany');
insert into orders values (10280, 'bergs', 2, '1996-08-14', '1996-09-11', '1996-09-12', 1, 8.97999954, 'berglunds snabbköp', 'berguvsvägen  8', 'luleå', null, 's-958 22', 'sweden');
insert into orders values (10281, 'romey', 4, '1996-08-14', '1996-08-28', '1996-08-21', 1, 2.94000006, 'romero y tomillo', 'gran vía, 1', 'madrid', null, '28001', 'spain');
insert into orders values (10282, 'romey', 4, '1996-08-15', '1996-09-12', '1996-08-21', 1, 12.6899996, 'romero y tomillo', 'gran vía, 1', 'madrid', null, '28001', 'spain');
insert into orders values (10283, 'lilas', 3, '1996-08-16', '1996-09-13', '1996-08-23', 3, 84.8099976, 'lila-supermercado', 'carrera 52 con ave. bolívar #65-98 llano largo', 'barquisimeto', 'lara', '3508', 'venezuela');
insert into orders values (10284, 'lehms', 4, '1996-08-19', '1996-09-16', '1996-08-27', 1, 76.5599976, 'lehmanns marktstand', 'magazinweg 7', 'frankfurt a.m.', null, '60528', 'germany');
insert into orders values (10285, 'quick', 1, '1996-08-20', '1996-09-17', '1996-08-26', 2, 76.8300018, 'quick-stop', 'taucherstraße 10', 'cunewalde', null, '01307', 'germany');
insert into orders values (10286, 'quick', 8, '1996-08-21', '1996-09-18', '1996-08-30', 3, 229.240005, 'quick-stop', 'taucherstraße 10', 'cunewalde', null, '01307', 'germany');
insert into orders values (10287, 'ricar', 8, '1996-08-22', '1996-09-19', '1996-08-28', 3, 12.7600002, 'ricardo adocicados', 'av. copacabana, 267', 'rio de janeiro', 'rj', '02389-890', 'brazil');
insert into orders values (10288, 'reggc', 4, '1996-08-23', '1996-09-20', '1996-09-03', 1, 7.44999981, 'reggiani caseifici', 'strada provinciale 124', 'reggio emilia', null, '42100', 'italy');
insert into orders values (10289, 'bsbev', 7, '1996-08-26', '1996-09-23', '1996-08-28', 3, 22.7700005, 'b''s beverages', 'fauntleroy circus', 'london', null, 'ec2 5nt', 'uk');
insert into orders values (10290, 'commi', 8, '1996-08-27', '1996-09-24', '1996-09-03', 1, 79.6999969, 'comércio mineiro', 'av. dos lusíadas, 23', 'sao paulo', 'sp', '05432-043', 'brazil');
insert into orders values (10291, 'quede', 6, '1996-08-27', '1996-09-24', '1996-09-04', 2, 6.4000001, 'que delícia', 'rua da panificadora, 12', 'rio de janeiro', 'rj', '02389-673', 'brazil');
insert into orders values (10292, 'tradh', 1, '1996-08-28', '1996-09-25', '1996-09-02', 2, 1.35000002, 'tradiçao hipermercados', 'av. inês de castro, 414', 'sao paulo', 'sp', '05634-030', 'brazil');
insert into orders values (10293, 'tortu', 1, '1996-08-29', '1996-09-26', '1996-09-11', 3, 21.1800003, 'tortuga restaurante', 'avda. azteca 123', 'méxico d.f.', null, '05033', 'mexico');
insert into orders values (10294, 'rattc', 4, '1996-08-30', '1996-09-27', '1996-09-05', 2, 147.259995, 'rattlesnake canyon grocery', '2817 milton dr.', 'albuquerque', 'nm', '87110', 'usa');
insert into orders values (10295, 'vinet', 2, '1996-09-02', '1996-09-30', '1996-09-10', 2, 1.14999998, 'vins et alcools chevalier', '59 rue de l''abbaye', 'reims', null, '51100', 'france');
insert into orders values (10296, 'lilas', 6, '1996-09-03', '1996-10-01', '1996-09-11', 1, 0.119999997, 'lila-supermercado', 'carrera 52 con ave. bolívar #65-98 llano largo', 'barquisimeto', 'lara', '3508', 'venezuela');
insert into orders values (10297, 'blonp', 5, '1996-09-04', '1996-10-16', '1996-09-10', 2, 5.73999977, 'blondel père et fils', '24, place kléber', 'strasbourg', null, '67000', 'france');
insert into orders values (10298, 'hungo', 6, '1996-09-05', '1996-10-03', '1996-09-11', 2, 168.220001, 'hungry owl all-night grocers', '8 johnstown road', 'cork', 'co. cork', null, 'ireland');
insert into orders values (10299, 'ricar', 4, '1996-09-06', '1996-10-04', '1996-09-13', 2, 29.7600002, 'ricardo adocicados', 'av. copacabana, 267', 'rio de janeiro', 'rj', '02389-890', 'brazil');
insert into orders values (10300, 'magaa', 2, '1996-09-09', '1996-10-07', '1996-09-18', 2, 17.6800003, 'magazzini alimentari riuniti', 'via ludovico il moro 22', 'bergamo', null, '24100', 'italy');
insert into orders values (10301, 'wandk', 8, '1996-09-09', '1996-10-07', '1996-09-17', 2, 45.0800018, 'die wandernde kuh', 'adenauerallee 900', 'stuttgart', null, '70563', 'germany');
insert into orders values (10302, 'suprd', 4, '1996-09-10', '1996-10-08', '1996-10-09', 2, 6.26999998, 'suprêmes délices', 'boulevard tirou, 255', 'charleroi', null, 'b-6000', 'belgium');
insert into orders values (10303, 'godos', 7, '1996-09-11', '1996-10-09', '1996-09-18', 2, 107.830002, 'godos cocina típica', 'c/ romero, 33', 'sevilla', null, '41101', 'spain');
insert into orders values (10304, 'tortu', 1, '1996-09-12', '1996-10-10', '1996-09-17', 2, 63.7900009, 'tortuga restaurante', 'avda. azteca 123', 'méxico d.f.', null, '05033', 'mexico');
insert into orders values (10305, 'oldwo', 8, '1996-09-13', '1996-10-11', '1996-10-09', 3, 257.619995, 'old world delicatessen', '2743 bering st.', 'anchorage', 'ak', '99508', 'usa');
insert into orders values (10306, 'romey', 1, '1996-09-16', '1996-10-14', '1996-09-23', 3, 7.55999994, 'romero y tomillo', 'gran vía, 1', 'madrid', null, '28001', 'spain');
insert into orders values (10307, 'lonep', 2, '1996-09-17', '1996-10-15', '1996-09-25', 2, 0.560000002, 'lonesome pine restaurant', '89 chiaroscuro rd.', 'portland', 'or', '97219', 'usa');
insert into orders values (10308, 'anatr', 7, '1996-09-18', '1996-10-16', '1996-09-24', 3, 1.61000001, 'ana trujillo emparedados y helados', 'avda. de la constitución 2222', 'méxico d.f.', null, '05021', 'mexico');
insert into orders values (10309, 'hungo', 3, '1996-09-19', '1996-10-17', '1996-10-23', 1, 47.2999992, 'hungry owl all-night grocers', '8 johnstown road', 'cork', 'co. cork', null, 'ireland');
insert into orders values (10310, 'thebi', 8, '1996-09-20', '1996-10-18', '1996-09-27', 2, 17.5200005, 'the big cheese', '89 jefferson way suite 2', 'portland', 'or', '97201', 'usa');
insert into orders values (10311, 'dumon', 1, '1996-09-20', '1996-10-04', '1996-09-26', 3, 24.6900005, 'du monde entier', '67, rue des cinquante otages', 'nantes', null, '44000', 'france');
insert into orders values (10312, 'wandk', 2, '1996-09-23', '1996-10-21', '1996-10-03', 2, 40.2599983, 'die wandernde kuh', 'adenauerallee 900', 'stuttgart', null, '70563', 'germany');
insert into orders values (10313, 'quick', 2, '1996-09-24', '1996-10-22', '1996-10-04', 2, 1.96000004, 'quick-stop', 'taucherstraße 10', 'cunewalde', null, '01307', 'germany');
insert into orders values (10314, 'rattc', 1, '1996-09-25', '1996-10-23', '1996-10-04', 2, 74.1600037, 'rattlesnake canyon grocery', '2817 milton dr.', 'albuquerque', 'nm', '87110', 'usa');
insert into orders values (10315, 'islat', 4, '1996-09-26', '1996-10-24', '1996-10-03', 2, 41.7599983, 'island trading', 'garden house crowther way', 'cowes', 'isle of wight', 'po31 7pj', 'uk');
insert into orders values (10316, 'rattc', 1, '1996-09-27', '1996-10-25', '1996-10-08', 3, 150.149994, 'rattlesnake canyon grocery', '2817 milton dr.', 'albuquerque', 'nm', '87110', 'usa');
insert into orders values (10317, 'lonep', 6, '1996-09-30', '1996-10-28', '1996-10-10', 1, 12.6899996, 'lonesome pine restaurant', '89 chiaroscuro rd.', 'portland', 'or', '97219', 'usa');
insert into orders values (10318, 'islat', 8, '1996-10-01', '1996-10-29', '1996-10-04', 2, 4.73000002, 'island trading', 'garden house crowther way', 'cowes', 'isle of wight', 'po31 7pj', 'uk');
insert into orders values (10319, 'tortu', 7, '1996-10-02', '1996-10-30', '1996-10-11', 3, 64.5, 'tortuga restaurante', 'avda. azteca 123', 'méxico d.f.', null, '05033', 'mexico');
insert into orders values (10320, 'warth', 5, '1996-10-03', '1996-10-17', '1996-10-18', 3, 34.5699997, 'wartian herkku', 'torikatu 38', 'oulu', null, '90110', 'finland');
insert into orders values (10321, 'islat', 3, '1996-10-03', '1996-10-31', '1996-10-11', 2, 3.43000007, 'island trading', 'garden house crowther way', 'cowes', 'isle of wight', 'po31 7pj', 'uk');
insert into orders values (10322, 'peric', 7, '1996-10-04', '1996-11-01', '1996-10-23', 3, 0.400000006, 'pericles comidas clásicas', 'calle dr. jorge cash 321', 'méxico d.f.', null, '05033', 'mexico');
insert into orders values (10323, 'koene', 4, '1996-10-07', '1996-11-04', '1996-10-14', 1, 4.88000011, 'königlich essen', 'maubelstr. 90', 'brandenburg', null, '14776', 'germany');
insert into orders values (10324, 'savea', 9, '1996-10-08', '1996-11-05', '1996-10-10', 1, 214.270004, 'save-a-lot markets', '187 suffolk ln.', 'boise', 'id', '83720', 'usa');
insert into orders values (10325, 'koene', 1, '1996-10-09', '1996-10-23', '1996-10-14', 3, 64.8600006, 'königlich essen', 'maubelstr. 90', 'brandenburg', null, '14776', 'germany');
insert into orders values (10326, 'bolid', 4, '1996-10-10', '1996-11-07', '1996-10-14', 2, 77.9199982, 'bólido comidas preparadas', 'c/ araquil, 67', 'madrid', null, '28023', 'spain');
insert into orders values (10327, 'folko', 2, '1996-10-11', '1996-11-08', '1996-10-14', 1, 63.3600006, 'folk och fä hb', 'åkergatan 24', 'bräcke', null, 's-844 67', 'sweden');
insert into orders values (10328, 'furib', 4, '1996-10-14', '1996-11-11', '1996-10-17', 3, 87.0299988, 'furia bacalhau e frutos do mar', 'jardim das rosas n. 32', 'lisboa', null, '1675', 'portugal');
insert into orders values (10329, 'splir', 4, '1996-10-15', '1996-11-26', '1996-10-23', 2, 191.669998, 'split rail beer & ale', 'p.o. box 555', 'lander', 'wy', '82520', 'usa');
insert into orders values (10330, 'lilas', 3, '1996-10-16', '1996-11-13', '1996-10-28', 1, 12.75, 'lila-supermercado', 'carrera 52 con ave. bolívar #65-98 llano largo', 'barquisimeto', 'lara', '3508', 'venezuela');
insert into orders values (10331, 'bonap', 9, '1996-10-16', '1996-11-27', '1996-10-21', 1, 10.1899996, 'bon app''', '12, rue des bouchers', 'marseille', null, '13008', 'france');
insert into orders values (10332, 'merep', 3, '1996-10-17', '1996-11-28', '1996-10-21', 2, 52.8400002, 'mère paillarde', '43 rue st. laurent', 'montréal', 'québec', 'h1j 1c3', 'canada');
insert into orders values (10333, 'warth', 5, '1996-10-18', '1996-11-15', '1996-10-25', 3, 0.589999974, 'wartian herkku', 'torikatu 38', 'oulu', null, '90110', 'finland');
insert into orders values (10334, 'victe', 8, '1996-10-21', '1996-11-18', '1996-10-28', 2, 8.56000042, 'victuailles en stock', '2, rue du commerce', 'lyon', null, '69004', 'france');
insert into orders values (10335, 'hungo', 7, '1996-10-22', '1996-11-19', '1996-10-24', 2, 42.1100006, 'hungry owl all-night grocers', '8 johnstown road', 'cork', 'co. cork', null, 'ireland');
insert into orders values (10336, 'prini', 7, '1996-10-23', '1996-11-20', '1996-10-25', 2, 15.5100002, 'princesa isabel vinhos', 'estrada da saúde n. 58', 'lisboa', null, '1756', 'portugal');
insert into orders values (10337, 'frank', 4, '1996-10-24', '1996-11-21', '1996-10-29', 3, 108.260002, 'frankenversand', 'berliner platz 43', 'münchen', null, '80805', 'germany');
insert into orders values (10338, 'oldwo', 4, '1996-10-25', '1996-11-22', '1996-10-29', 3, 84.2099991, 'old world delicatessen', '2743 bering st.', 'anchorage', 'ak', '99508', 'usa');
insert into orders values (10339, 'merep', 2, '1996-10-28', '1996-11-25', '1996-11-04', 2, 15.6599998, 'mère paillarde', '43 rue st. laurent', 'montréal', 'québec', 'h1j 1c3', 'canada');
insert into orders values (10340, 'bonap', 1, '1996-10-29', '1996-11-26', '1996-11-08', 3, 166.309998, 'bon app''', '12, rue des bouchers', 'marseille', null, '13008', 'france');
insert into orders values (10341, 'simob', 7, '1996-10-29', '1996-11-26', '1996-11-05', 3, 26.7800007, 'simons bistro', 'vinbæltet 34', 'kobenhavn', null, '1734', 'denmark');
insert into orders values (10342, 'frank', 4, '1996-10-30', '1996-11-13', '1996-11-04', 2, 54.8300018, 'frankenversand', 'berliner platz 43', 'münchen', null, '80805', 'germany');
insert into orders values (10343, 'lehms', 4, '1996-10-31', '1996-11-28', '1996-11-06', 1, 110.370003, 'lehmanns marktstand', 'magazinweg 7', 'frankfurt a.m.', null, '60528', 'germany');
insert into orders values (10344, 'whitc', 4, '1996-11-01', '1996-11-29', '1996-11-05', 2, 23.2900009, 'white clover markets', '1029 - 12th ave. s.', 'seattle', 'wa', '98124', 'usa');
insert into orders values (10345, 'quick', 2, '1996-11-04', '1996-12-02', '1996-11-11', 2, 249.059998, 'quick-stop', 'taucherstraße 10', 'cunewalde', null, '01307', 'germany');
insert into orders values (10346, 'rattc', 3, '1996-11-05', '1996-12-17', '1996-11-08', 3, 142.080002, 'rattlesnake canyon grocery', '2817 milton dr.', 'albuquerque', 'nm', '87110', 'usa');
insert into orders values (10347, 'famia', 4, '1996-11-06', '1996-12-04', '1996-11-08', 3, 3.0999999, 'familia arquibaldo', 'rua orós, 92', 'sao paulo', 'sp', '05442-030', 'brazil');
insert into orders values (10348, 'wandk', 4, '1996-11-07', '1996-12-05', '1996-11-15', 2, 0.779999971, 'die wandernde kuh', 'adenauerallee 900', 'stuttgart', null, '70563', 'germany');
insert into orders values (10349, 'splir', 7, '1996-11-08', '1996-12-06', '1996-11-15', 1, 8.63000011, 'split rail beer & ale', 'p.o. box 555', 'lander', 'wy', '82520', 'usa');
insert into orders values (10350, 'lamai', 6, '1996-11-11', '1996-12-09', '1996-12-03', 2, 64.1900024, 'la maison d''asie', '1 rue alsace-lorraine', 'toulouse', null, '31000', 'france');
insert into orders values (10351, 'ernsh', 1, '1996-11-11', '1996-12-09', '1996-11-20', 1, 162.330002, 'ernst handel', 'kirchgasse 6', 'graz', null, '8010', 'austria');
insert into orders values (10352, 'furib', 3, '1996-11-12', '1996-11-26', '1996-11-18', 3, 1.29999995, 'furia bacalhau e frutos do mar', 'jardim das rosas n. 32', 'lisboa', null, '1675', 'portugal');
insert into orders values (10353, 'picco', 7, '1996-11-13', '1996-12-11', '1996-11-25', 3, 360.630005, 'piccolo und mehr', 'geislweg 14', 'salzburg', null, '5020', 'austria');
insert into orders values (10354, 'peric', 8, '1996-11-14', '1996-12-12', '1996-11-20', 3, 53.7999992, 'pericles comidas clásicas', 'calle dr. jorge cash 321', 'méxico d.f.', null, '05033', 'mexico');
insert into orders values (10355, 'arout', 6, '1996-11-15', '1996-12-13', '1996-11-20', 1, 41.9500008, 'around the horn', 'brook farm stratford st. mary', 'colchester', 'essex', 'co7 6jx', 'uk');
insert into orders values (10356, 'wandk', 6, '1996-11-18', '1996-12-16', '1996-11-27', 2, 36.7099991, 'die wandernde kuh', 'adenauerallee 900', 'stuttgart', null, '70563', 'germany');
insert into orders values (10357, 'lilas', 1, '1996-11-19', '1996-12-17', '1996-12-02', 3, 34.8800011, 'lila-supermercado', 'carrera 52 con ave. bolívar #65-98 llano largo', 'barquisimeto', 'lara', '3508', 'venezuela');
insert into orders values (10358, 'lamai', 5, '1996-11-20', '1996-12-18', '1996-11-27', 1, 19.6399994, 'la maison d''asie', '1 rue alsace-lorraine', 'toulouse', null, '31000', 'france');
insert into orders values (10359, 'seves', 5, '1996-11-21', '1996-12-19', '1996-11-26', 3, 288.429993, 'seven seas imports', '90 wadhurst rd.', 'london', null, 'ox15 4nb', 'uk');
insert into orders values (10360, 'blonp', 4, '1996-11-22', '1996-12-20', '1996-12-02', 3, 131.699997, 'blondel père et fils', '24, place kléber', 'strasbourg', null, '67000', 'france');
insert into orders values (10361, 'quick', 1, '1996-11-22', '1996-12-20', '1996-12-03', 2, 183.169998, 'quick-stop', 'taucherstraße 10', 'cunewalde', null, '01307', 'germany');
insert into orders values (10362, 'bonap', 3, '1996-11-25', '1996-12-23', '1996-11-28', 1, 96.0400009, 'bon app''', '12, rue des bouchers', 'marseille', null, '13008', 'france');
insert into orders values (10363, 'dracd', 4, '1996-11-26', '1996-12-24', '1996-12-04', 3, 30.5400009, 'drachenblut delikatessen', 'walserweg 21', 'aachen', null, '52066', 'germany');
insert into orders values (10364, 'eastc', 1, '1996-11-26', '1997-01-07', '1996-12-04', 1, 71.9700012, 'eastern connection', '35 king george', 'london', null, 'wx3 6fw', 'uk');
insert into orders values (10365, 'anton', 3, '1996-11-27', '1996-12-25', '1996-12-02', 2, 22, 'antonio moreno taquería', 'mataderos  2312', 'méxico d.f.', null, '05023', 'mexico');
insert into orders values (10366, 'galed', 8, '1996-11-28', '1997-01-09', '1996-12-30', 2, 10.1400003, 'galería del gastronómo', 'rambla de cataluña, 23', 'barcelona', null, '8022', 'spain');
insert into orders values (10367, 'vaffe', 7, '1996-11-28', '1996-12-26', '1996-12-02', 3, 13.5500002, 'vaffeljernet', 'smagsloget 45', 'århus', null, '8200', 'denmark');
insert into orders values (10368, 'ernsh', 2, '1996-11-29', '1996-12-27', '1996-12-02', 2, 101.949997, 'ernst handel', 'kirchgasse 6', 'graz', null, '8010', 'austria');
insert into orders values (10369, 'splir', 8, '1996-12-02', '1996-12-30', '1996-12-09', 2, 195.679993, 'split rail beer & ale', 'p.o. box 555', 'lander', 'wy', '82520', 'usa');
insert into orders values (10370, 'chops', 6, '1996-12-03', '1996-12-31', '1996-12-27', 2, 1.16999996, 'chop-suey chinese', 'hauptstr. 31', 'bern', null, '3012', 'switzerland');
insert into orders values (10371, 'lamai', 1, '1996-12-03', '1996-12-31', '1996-12-24', 1, 0.449999988, 'la maison d''asie', '1 rue alsace-lorraine', 'toulouse', null, '31000', 'france');
insert into orders values (10372, 'queen', 5, '1996-12-04', '1997-01-01', '1996-12-09', 2, 890.780029, 'queen cozinha', 'alameda dos canàrios, 891', 'sao paulo', 'sp', '05487-020', 'brazil');
insert into orders values (10373, 'hungo', 4, '1996-12-05', '1997-01-02', '1996-12-11', 3, 124.120003, 'hungry owl all-night grocers', '8 johnstown road', 'cork', 'co. cork', null, 'ireland');
insert into orders values (10374, 'wolza', 1, '1996-12-05', '1997-01-02', '1996-12-09', 3, 3.94000006, 'wolski zajazd', 'ul. filtrowa 68', 'warszawa', null, '01-012', 'poland');
insert into orders values (10375, 'hungc', 3, '1996-12-06', '1997-01-03', '1996-12-09', 2, 20.1200008, 'hungry coyote import store', 'city center plaza 516 main st.', 'elgin', 'or', '97827', 'usa');
insert into orders values (10376, 'merep', 1, '1996-12-09', '1997-01-06', '1996-12-13', 2, 20.3899994, 'mère paillarde', '43 rue st. laurent', 'montréal', 'québec', 'h1j 1c3', 'canada');
insert into orders values (10377, 'seves', 1, '1996-12-09', '1997-01-06', '1996-12-13', 3, 22.2099991, 'seven seas imports', '90 wadhurst rd.', 'london', null, 'ox15 4nb', 'uk');
insert into orders values (10378, 'folko', 5, '1996-12-10', '1997-01-07', '1996-12-19', 3, 5.44000006, 'folk och fä hb', 'åkergatan 24', 'bräcke', null, 's-844 67', 'sweden');
insert into orders values (10379, 'quede', 2, '1996-12-11', '1997-01-08', '1996-12-13', 1, 45.0299988, 'que delícia', 'rua da panificadora, 12', 'rio de janeiro', 'rj', '02389-673', 'brazil');
insert into orders values (10380, 'hungo', 8, '1996-12-12', '1997-01-09', '1997-01-16', 3, 35.0299988, 'hungry owl all-night grocers', '8 johnstown road', 'cork', 'co. cork', null, 'ireland');
insert into orders values (10381, 'lilas', 3, '1996-12-12', '1997-01-09', '1996-12-13', 3, 7.98999977, 'lila-supermercado', 'carrera 52 con ave. bolívar #65-98 llano largo', 'barquisimeto', 'lara', '3508', 'venezuela');
insert into orders values (10382, 'ernsh', 4, '1996-12-13', '1997-01-10', '1996-12-16', 1, 94.7699966, 'ernst handel', 'kirchgasse 6', 'graz', null, '8010', 'austria');
insert into orders values (10383, 'arout', 8, '1996-12-16', '1997-01-13', '1996-12-18', 3, 34.2400017, 'around the horn', 'brook farm stratford st. mary', 'colchester', 'essex', 'co7 6jx', 'uk');
insert into orders values (10384, 'bergs', 3, '1996-12-16', '1997-01-13', '1996-12-20', 3, 168.639999, 'berglunds snabbköp', 'berguvsvägen  8', 'luleå', null, 's-958 22', 'sweden');
insert into orders values (10385, 'splir', 1, '1996-12-17', '1997-01-14', '1996-12-23', 2, 30.9599991, 'split rail beer & ale', 'p.o. box 555', 'lander', 'wy', '82520', 'usa');
insert into orders values (10386, 'famia', 9, '1996-12-18', '1997-01-01', '1996-12-25', 3, 13.9899998, 'familia arquibaldo', 'rua orós, 92', 'sao paulo', 'sp', '05442-030', 'brazil');
insert into orders values (10387, 'santg', 1, '1996-12-18', '1997-01-15', '1996-12-20', 2, 93.6299973, 'santé gourmet', 'erling skakkes gate 78', 'stavern', null, '4110', 'norway');
insert into orders values (10388, 'seves', 2, '1996-12-19', '1997-01-16', '1996-12-20', 1, 34.8600006, 'seven seas imports', '90 wadhurst rd.', 'london', null, 'ox15 4nb', 'uk');
insert into orders values (10389, 'bottm', 4, '1996-12-20', '1997-01-17', '1996-12-24', 2, 47.4199982, 'bottom-dollar markets', '23 tsawassen blvd.', 'tsawassen', 'bc', 't2f 8m4', 'canada');
insert into orders values (10390, 'ernsh', 6, '1996-12-23', '1997-01-20', '1996-12-26', 1, 126.379997, 'ernst handel', 'kirchgasse 6', 'graz', null, '8010', 'austria');
insert into orders values (10391, 'dracd', 3, '1996-12-23', '1997-01-20', '1996-12-31', 3, 5.44999981, 'drachenblut delikatessen', 'walserweg 21', 'aachen', null, '52066', 'germany');
insert into orders values (10392, 'picco', 2, '1996-12-24', '1997-01-21', '1997-01-01', 3, 122.459999, 'piccolo und mehr', 'geislweg 14', 'salzburg', null, '5020', 'austria');
insert into orders values (10393, 'savea', 1, '1996-12-25', '1997-01-22', '1997-01-03', 3, 126.559998, 'save-a-lot markets', '187 suffolk ln.', 'boise', 'id', '83720', 'usa');
insert into orders values (10394, 'hungc', 1, '1996-12-25', '1997-01-22', '1997-01-03', 3, 30.3400002, 'hungry coyote import store', 'city center plaza 516 main st.', 'elgin', 'or', '97827', 'usa');
insert into orders values (10395, 'hilaa', 6, '1996-12-26', '1997-01-23', '1997-01-03', 1, 184.410004, 'hilarion-abastos', 'carrera 22 con ave. carlos soublette #8-35', 'san cristóbal', 'táchira', '5022', 'venezuela');
insert into orders values (10396, 'frank', 1, '1996-12-27', '1997-01-10', '1997-01-06', 3, 135.350006, 'frankenversand', 'berliner platz 43', 'münchen', null, '80805', 'germany');
insert into orders values (10397, 'prini', 5, '1996-12-27', '1997-01-24', '1997-01-02', 1, 60.2599983, 'princesa isabel vinhos', 'estrada da saúde n. 58', 'lisboa', null, '1756', 'portugal');
insert into orders values (10398, 'savea', 2, '1996-12-30', '1997-01-27', '1997-01-09', 3, 89.1600037, 'save-a-lot markets', '187 suffolk ln.', 'boise', 'id', '83720', 'usa');
insert into orders values (10399, 'vaffe', 8, '1996-12-31', '1997-01-14', '1997-01-08', 3, 27.3600006, 'vaffeljernet', 'smagsloget 45', 'århus', null, '8200', 'denmark');
insert into orders values (10400, 'eastc', 1, '1997-01-01', '1997-01-29', '1997-01-16', 3, 83.9300003, 'eastern connection', '35 king george', 'london', null, 'wx3 6fw', 'uk');
insert into orders values (10401, 'rattc', 1, '1997-01-01', '1997-01-29', '1997-01-10', 1, 12.5100002, 'rattlesnake canyon grocery', '2817 milton dr.', 'albuquerque', 'nm', '87110', 'usa');
insert into orders values (10402, 'ernsh', 8, '1997-01-02', '1997-02-13', '1997-01-10', 2, 67.8799973, 'ernst handel', 'kirchgasse 6', 'graz', null, '8010', 'austria');
insert into orders values (10403, 'ernsh', 4, '1997-01-03', '1997-01-31', '1997-01-09', 3, 73.7900009, 'ernst handel', 'kirchgasse 6', 'graz', null, '8010', 'austria');
insert into orders values (10404, 'magaa', 2, '1997-01-03', '1997-01-31', '1997-01-08', 1, 155.970001, 'magazzini alimentari riuniti', 'via ludovico il moro 22', 'bergamo', null, '24100', 'italy');
insert into orders values (10405, 'linod', 1, '1997-01-06', '1997-02-03', '1997-01-22', 1, 34.8199997, 'lino-delicateses', 'ave. 5 de mayo porlamar', 'i. de margarita', 'nueva esparta', '4980', 'venezuela');
insert into orders values (10406, 'queen', 7, '1997-01-07', '1997-02-18', '1997-01-13', 1, 108.040001, 'queen cozinha', 'alameda dos canàrios, 891', 'sao paulo', 'sp', '05487-020', 'brazil');
insert into orders values (10407, 'ottik', 2, '1997-01-07', '1997-02-04', '1997-01-30', 2, 91.4800034, 'ottilies käseladen', 'mehrheimerstr. 369', 'köln', null, '50739', 'germany');
insert into orders values (10408, 'folig', 8, '1997-01-08', '1997-02-05', '1997-01-14', 1, 11.2600002, 'folies gourmandes', '184, chaussée de tournai', 'lille', null, '59000', 'france');
insert into orders values (10409, 'ocean', 3, '1997-01-09', '1997-02-06', '1997-01-14', 1, 29.8299999, 'océano atlántico ltda.', 'ing. gustavo moncada 8585 piso 20-a', 'buenos aires', null, '1010', 'argentina');
insert into orders values (10410, 'bottm', 3, '1997-01-10', '1997-02-07', '1997-01-15', 3, 2.4000001, 'bottom-dollar markets', '23 tsawassen blvd.', 'tsawassen', 'bc', 't2f 8m4', 'canada');
insert into orders values (10411, 'bottm', 9, '1997-01-10', '1997-02-07', '1997-01-21', 3, 23.6499996, 'bottom-dollar markets', '23 tsawassen blvd.', 'tsawassen', 'bc', 't2f 8m4', 'canada');
insert into orders values (10412, 'warth', 8, '1997-01-13', '1997-02-10', '1997-01-15', 2, 3.76999998, 'wartian herkku', 'torikatu 38', 'oulu', null, '90110', 'finland');
insert into orders values (10413, 'lamai', 3, '1997-01-14', '1997-02-11', '1997-01-16', 2, 95.6600037, 'la maison d''asie', '1 rue alsace-lorraine', 'toulouse', null, '31000', 'france');
insert into orders values (10414, 'famia', 2, '1997-01-14', '1997-02-11', '1997-01-17', 3, 21.4799995, 'familia arquibaldo', 'rua orós, 92', 'sao paulo', 'sp', '05442-030', 'brazil');
insert into orders values (10415, 'hungc', 3, '1997-01-15', '1997-02-12', '1997-01-24', 1, 0.200000003, 'hungry coyote import store', 'city center plaza 516 main st.', 'elgin', 'or', '97827', 'usa');
insert into orders values (10416, 'warth', 8, '1997-01-16', '1997-02-13', '1997-01-27', 3, 22.7199993, 'wartian herkku', 'torikatu 38', 'oulu', null, '90110', 'finland');
insert into orders values (10417, 'simob', 4, '1997-01-16', '1997-02-13', '1997-01-28', 3, 70.2900009, 'simons bistro', 'vinbæltet 34', 'kobenhavn', null, '1734', 'denmark');
insert into orders values (10418, 'quick', 4, '1997-01-17', '1997-02-14', '1997-01-24', 1, 17.5499992, 'quick-stop', 'taucherstraße 10', 'cunewalde', null, '01307', 'germany');
insert into orders values (10419, 'ricsu', 4, '1997-01-20', '1997-02-17', '1997-01-30', 2, 137.350006, 'richter supermarkt', 'starenweg 5', 'genève', null, '1204', 'switzerland');
insert into orders values (10420, 'welli', 3, '1997-01-21', '1997-02-18', '1997-01-27', 1, 44.1199989, 'wellington importadora', 'rua do mercado, 12', 'resende', 'sp', '08737-363', 'brazil');
insert into orders values (10421, 'quede', 8, '1997-01-21', '1997-03-04', '1997-01-27', 1, 99.2300034, 'que delícia', 'rua da panificadora, 12', 'rio de janeiro', 'rj', '02389-673', 'brazil');
insert into orders values (10422, 'frans', 2, '1997-01-22', '1997-02-19', '1997-01-31', 1, 3.01999998, 'franchi s.p.a.', 'via monte bianco 34', 'torino', null, '10100', 'italy');
insert into orders values (10423, 'gourl', 6, '1997-01-23', '1997-02-06', '1997-02-24', 3, 24.5, 'gourmet lanchonetes', 'av. brasil, 442', 'campinas', 'sp', '04876-786', 'brazil');
insert into orders values (10424, 'merep', 7, '1997-01-23', '1997-02-20', '1997-01-27', 2, 370.609985, 'mère paillarde', '43 rue st. laurent', 'montréal', 'québec', 'h1j 1c3', 'canada');
insert into orders values (10425, 'lamai', 6, '1997-01-24', '1997-02-21', '1997-02-14', 2, 7.92999983, 'la maison d''asie', '1 rue alsace-lorraine', 'toulouse', null, '31000', 'france');
insert into orders values (10426, 'galed', 4, '1997-01-27', '1997-02-24', '1997-02-06', 1, 18.6900005, 'galería del gastronómo', 'rambla de cataluña, 23', 'barcelona', null, '8022', 'spain');
insert into orders values (10427, 'picco', 4, '1997-01-27', '1997-02-24', '1997-03-03', 2, 31.2900009, 'piccolo und mehr', 'geislweg 14', 'salzburg', null, '5020', 'austria');
insert into orders values (10428, 'reggc', 7, '1997-01-28', '1997-02-25', '1997-02-04', 1, 11.0900002, 'reggiani caseifici', 'strada provinciale 124', 'reggio emilia', null, '42100', 'italy');
insert into orders values (10429, 'hungo', 3, '1997-01-29', '1997-03-12', '1997-02-07', 2, 56.6300011, 'hungry owl all-night grocers', '8 johnstown road', 'cork', 'co. cork', null, 'ireland');
insert into orders values (10430, 'ernsh', 4, '1997-01-30', '1997-02-13', '1997-02-03', 1, 458.779999, 'ernst handel', 'kirchgasse 6', 'graz', null, '8010', 'austria');
insert into orders values (10431, 'bottm', 4, '1997-01-30', '1997-02-13', '1997-02-07', 2, 44.1699982, 'bottom-dollar markets', '23 tsawassen blvd.', 'tsawassen', 'bc', 't2f 8m4', 'canada');
insert into orders values (10432, 'splir', 3, '1997-01-31', '1997-02-14', '1997-02-07', 2, 4.34000015, 'split rail beer & ale', 'p.o. box 555', 'lander', 'wy', '82520', 'usa');
insert into orders values (10433, 'prini', 3, '1997-02-03', '1997-03-03', '1997-03-04', 3, 73.8300018, 'princesa isabel vinhos', 'estrada da saúde n. 58', 'lisboa', null, '1756', 'portugal');
insert into orders values (10434, 'folko', 3, '1997-02-03', '1997-03-03', '1997-02-13', 2, 17.9200001, 'folk och fä hb', 'åkergatan 24', 'bräcke', null, 's-844 67', 'sweden');
insert into orders values (10435, 'consh', 8, '1997-02-04', '1997-03-18', '1997-02-07', 2, 9.21000004, 'consolidated holdings', 'berkeley gardens 12  brewery', 'london', null, 'wx1 6lt', 'uk');
insert into orders values (10436, 'blonp', 3, '1997-02-05', '1997-03-05', '1997-02-11', 2, 156.660004, 'blondel père et fils', '24, place kléber', 'strasbourg', null, '67000', 'france');
insert into orders values (10437, 'warth', 8, '1997-02-05', '1997-03-05', '1997-02-12', 1, 19.9699993, 'wartian herkku', 'torikatu 38', 'oulu', null, '90110', 'finland');
insert into orders values (10438, 'tomsp', 3, '1997-02-06', '1997-03-06', '1997-02-14', 2, 8.23999977, 'toms spezialitäten', 'luisenstr. 48', 'münster', null, '44087', 'germany');
insert into orders values (10439, 'merep', 6, '1997-02-07', '1997-03-07', '1997-02-10', 3, 4.07000017, 'mère paillarde', '43 rue st. laurent', 'montréal', 'québec', 'h1j 1c3', 'canada');
insert into orders values (10440, 'savea', 4, '1997-02-10', '1997-03-10', '1997-02-28', 2, 86.5299988, 'save-a-lot markets', '187 suffolk ln.', 'boise', 'id', '83720', 'usa');
insert into orders values (10441, 'oldwo', 3, '1997-02-10', '1997-03-24', '1997-03-14', 2, 73.0199966, 'old world delicatessen', '2743 bering st.', 'anchorage', 'ak', '99508', 'usa');
insert into orders values (10442, 'ernsh', 3, '1997-02-11', '1997-03-11', '1997-02-18', 2, 47.9399986, 'ernst handel', 'kirchgasse 6', 'graz', null, '8010', 'austria');
insert into orders values (10443, 'reggc', 8, '1997-02-12', '1997-03-12', '1997-02-14', 1, 13.9499998, 'reggiani caseifici', 'strada provinciale 124', 'reggio emilia', null, '42100', 'italy');
insert into orders values (10444, 'bergs', 3, '1997-02-12', '1997-03-12', '1997-02-21', 3, 3.5, 'berglunds snabbköp', 'berguvsvägen  8', 'luleå', null, 's-958 22', 'sweden');
insert into orders values (10445, 'bergs', 3, '1997-02-13', '1997-03-13', '1997-02-20', 1, 9.30000019, 'berglunds snabbköp', 'berguvsvägen  8', 'luleå', null, 's-958 22', 'sweden');
insert into orders values (10446, 'tomsp', 6, '1997-02-14', '1997-03-14', '1997-02-19', 1, 14.6800003, 'toms spezialitäten', 'luisenstr. 48', 'münster', null, '44087', 'germany');
insert into orders values (10447, 'ricar', 4, '1997-02-14', '1997-03-14', '1997-03-07', 2, 68.6600037, 'ricardo adocicados', 'av. copacabana, 267', 'rio de janeiro', 'rj', '02389-890', 'brazil');
insert into orders values (10448, 'ranch', 4, '1997-02-17', '1997-03-17', '1997-02-24', 2, 38.8199997, 'rancho grande', 'av. del libertador 900', 'buenos aires', null, '1010', 'argentina');
insert into orders values (10449, 'blonp', 3, '1997-02-18', '1997-03-18', '1997-02-27', 2, 53.2999992, 'blondel père et fils', '24, place kléber', 'strasbourg', null, '67000', 'france');
insert into orders values (10450, 'victe', 8, '1997-02-19', '1997-03-19', '1997-03-11', 2, 7.23000002, 'victuailles en stock', '2, rue du commerce', 'lyon', null, '69004', 'france');
insert into orders values (10451, 'quick', 4, '1997-02-19', '1997-03-05', '1997-03-12', 3, 189.089996, 'quick-stop', 'taucherstraße 10', 'cunewalde', null, '01307', 'germany');
insert into orders values (10452, 'savea', 8, '1997-02-20', '1997-03-20', '1997-02-26', 1, 140.259995, 'save-a-lot markets', '187 suffolk ln.', 'boise', 'id', '83720', 'usa');
insert into orders values (10453, 'arout', 1, '1997-02-21', '1997-03-21', '1997-02-26', 2, 25.3600006, 'around the horn', 'brook farm stratford st. mary', 'colchester', 'essex', 'co7 6jx', 'uk');
insert into orders values (10454, 'lamai', 4, '1997-02-21', '1997-03-21', '1997-02-25', 3, 2.74000001, 'la maison d''asie', '1 rue alsace-lorraine', 'toulouse', null, '31000', 'france');
insert into orders values (10455, 'warth', 8, '1997-02-24', '1997-04-07', '1997-03-03', 2, 180.449997, 'wartian herkku', 'torikatu 38', 'oulu', null, '90110', 'finland');
insert into orders values (10456, 'koene', 8, '1997-02-25', '1997-04-08', '1997-02-28', 2, 8.11999989, 'königlich essen', 'maubelstr. 90', 'brandenburg', null, '14776', 'germany');
insert into orders values (10457, 'koene', 2, '1997-02-25', '1997-03-25', '1997-03-03', 1, 11.5699997, 'königlich essen', 'maubelstr. 90', 'brandenburg', null, '14776', 'germany');
insert into orders values (10458, 'suprd', 7, '1997-02-26', '1997-03-26', '1997-03-04', 3, 147.059998, 'suprêmes délices', 'boulevard tirou, 255', 'charleroi', null, 'b-6000', 'belgium');
insert into orders values (10459, 'victe', 4, '1997-02-27', '1997-03-27', '1997-02-28', 2, 25.0900002, 'victuailles en stock', '2, rue du commerce', 'lyon', null, '69004', 'france');
insert into orders values (10460, 'folko', 8, '1997-02-28', '1997-03-28', '1997-03-03', 1, 16.2700005, 'folk och fä hb', 'åkergatan 24', 'bräcke', null, 's-844 67', 'sweden');
insert into orders values (10461, 'lilas', 1, '1997-02-28', '1997-03-28', '1997-03-05', 3, 148.610001, 'lila-supermercado', 'carrera 52 con ave. bolívar #65-98 llano largo', 'barquisimeto', 'lara', '3508', 'venezuela');
insert into orders values (10462, 'consh', 2, '1997-03-03', '1997-03-31', '1997-03-18', 1, 6.17000008, 'consolidated holdings', 'berkeley gardens 12  brewery', 'london', null, 'wx1 6lt', 'uk');
insert into orders values (10463, 'suprd', 5, '1997-03-04', '1997-04-01', '1997-03-06', 3, 14.7799997, 'suprêmes délices', 'boulevard tirou, 255', 'charleroi', null, 'b-6000', 'belgium');
insert into orders values (10464, 'furib', 4, '1997-03-04', '1997-04-01', '1997-03-14', 2, 89, 'furia bacalhau e frutos do mar', 'jardim das rosas n. 32', 'lisboa', null, '1675', 'portugal');
insert into orders values (10465, 'vaffe', 1, '1997-03-05', '1997-04-02', '1997-03-14', 3, 145.039993, 'vaffeljernet', 'smagsloget 45', 'århus', null, '8200', 'denmark');
insert into orders values (10466, 'commi', 4, '1997-03-06', '1997-04-03', '1997-03-13', 1, 11.9300003, 'comércio mineiro', 'av. dos lusíadas, 23', 'sao paulo', 'sp', '05432-043', 'brazil');
insert into orders values (10467, 'magaa', 8, '1997-03-06', '1997-04-03', '1997-03-11', 2, 4.92999983, 'magazzini alimentari riuniti', 'via ludovico il moro 22', 'bergamo', null, '24100', 'italy');
insert into orders values (10468, 'koene', 3, '1997-03-07', '1997-04-04', '1997-03-12', 3, 44.1199989, 'königlich essen', 'maubelstr. 90', 'brandenburg', null, '14776', 'germany');
insert into orders values (10469, 'whitc', 1, '1997-03-10', '1997-04-07', '1997-03-14', 1, 60.1800003, 'white clover markets', '1029 - 12th ave. s.', 'seattle', 'wa', '98124', 'usa');
insert into orders values (10470, 'bonap', 4, '1997-03-11', '1997-04-08', '1997-03-14', 2, 64.5599976, 'bon app''', '12, rue des bouchers', 'marseille', null, '13008', 'france');
insert into orders values (10471, 'bsbev', 2, '1997-03-11', '1997-04-08', '1997-03-18', 3, 45.5900002, 'b''s beverages', 'fauntleroy circus', 'london', null, 'ec2 5nt', 'uk');
insert into orders values (10472, 'seves', 8, '1997-03-12', '1997-04-09', '1997-03-19', 1, 4.19999981, 'seven seas imports', '90 wadhurst rd.', 'london', null, 'ox15 4nb', 'uk');
insert into orders values (10473, 'islat', 1, '1997-03-13', '1997-03-27', '1997-03-21', 3, 16.3700008, 'island trading', 'garden house crowther way', 'cowes', 'isle of wight', 'po31 7pj', 'uk');
insert into orders values (10474, 'peric', 5, '1997-03-13', '1997-04-10', '1997-03-21', 2, 83.4899979, 'pericles comidas clásicas', 'calle dr. jorge cash 321', 'méxico d.f.', null, '05033', 'mexico');
insert into orders values (10475, 'suprd', 9, '1997-03-14', '1997-04-11', '1997-04-04', 1, 68.5199966, 'suprêmes délices', 'boulevard tirou, 255', 'charleroi', null, 'b-6000', 'belgium');
insert into orders values (10476, 'hilaa', 8, '1997-03-17', '1997-04-14', '1997-03-24', 3, 4.40999985, 'hilarion-abastos', 'carrera 22 con ave. carlos soublette #8-35', 'san cristóbal', 'táchira', '5022', 'venezuela');
insert into orders values (10477, 'prini', 5, '1997-03-17', '1997-04-14', '1997-03-25', 2, 13.0200005, 'princesa isabel vinhos', 'estrada da saúde n. 58', 'lisboa', null, '1756', 'portugal');
insert into orders values (10478, 'victe', 2, '1997-03-18', '1997-04-01', '1997-03-26', 3, 4.80999994, 'victuailles en stock', '2, rue du commerce', 'lyon', null, '69004', 'france');
insert into orders values (10479, 'rattc', 3, '1997-03-19', '1997-04-16', '1997-03-21', 3, 708.950012, 'rattlesnake canyon grocery', '2817 milton dr.', 'albuquerque', 'nm', '87110', 'usa');
insert into orders values (10480, 'folig', 6, '1997-03-20', '1997-04-17', '1997-03-24', 2, 1.35000002, 'folies gourmandes', '184, chaussée de tournai', 'lille', null, '59000', 'france');
insert into orders values (10481, 'ricar', 8, '1997-03-20', '1997-04-17', '1997-03-25', 2, 64.3300018, 'ricardo adocicados', 'av. copacabana, 267', 'rio de janeiro', 'rj', '02389-890', 'brazil');
insert into orders values (10482, 'lazyk', 1, '1997-03-21', '1997-04-18', '1997-04-10', 3, 7.48000002, 'lazy k kountry store', '12 orchestra terrace', 'walla walla', 'wa', '99362', 'usa');
insert into orders values (10483, 'whitc', 7, '1997-03-24', '1997-04-21', '1997-04-25', 2, 15.2799997, 'white clover markets', '1029 - 12th ave. s.', 'seattle', 'wa', '98124', 'usa');
insert into orders values (10484, 'bsbev', 3, '1997-03-24', '1997-04-21', '1997-04-01', 3, 6.88000011, 'b''s beverages', 'fauntleroy circus', 'london', null, 'ec2 5nt', 'uk');
insert into orders values (10485, 'linod', 4, '1997-03-25', '1997-04-08', '1997-03-31', 2, 64.4499969, 'lino-delicateses', 'ave. 5 de mayo porlamar', 'i. de margarita', 'nueva esparta', '4980', 'venezuela');
insert into orders values (10486, 'hilaa', 1, '1997-03-26', '1997-04-23', '1997-04-02', 2, 30.5300007, 'hilarion-abastos', 'carrera 22 con ave. carlos soublette #8-35', 'san cristóbal', 'táchira', '5022', 'venezuela');
insert into orders values (10487, 'queen', 2, '1997-03-26', '1997-04-23', '1997-03-28', 2, 71.0699997, 'queen cozinha', 'alameda dos canàrios, 891', 'sao paulo', 'sp', '05487-020', 'brazil');
insert into orders values (10488, 'frank', 8, '1997-03-27', '1997-04-24', '1997-04-02', 2, 4.92999983, 'frankenversand', 'berliner platz 43', 'münchen', null, '80805', 'germany');
insert into orders values (10489, 'picco', 6, '1997-03-28', '1997-04-25', '1997-04-09', 2, 5.28999996, 'piccolo und mehr', 'geislweg 14', 'salzburg', null, '5020', 'austria');
insert into orders values (10490, 'hilaa', 7, '1997-03-31', '1997-04-28', '1997-04-03', 2, 210.190002, 'hilarion-abastos', 'carrera 22 con ave. carlos soublette #8-35', 'san cristóbal', 'táchira', '5022', 'venezuela');
insert into orders values (10491, 'furib', 8, '1997-03-31', '1997-04-28', '1997-04-08', 3, 16.9599991, 'furia bacalhau e frutos do mar', 'jardim das rosas n. 32', 'lisboa', null, '1675', 'portugal');
insert into orders values (10492, 'bottm', 3, '1997-04-01', '1997-04-29', '1997-04-11', 1, 62.8899994, 'bottom-dollar markets', '23 tsawassen blvd.', 'tsawassen', 'bc', 't2f 8m4', 'canada');
insert into orders values (10493, 'lamai', 4, '1997-04-02', '1997-04-30', '1997-04-10', 3, 10.6400003, 'la maison d''asie', '1 rue alsace-lorraine', 'toulouse', null, '31000', 'france');
insert into orders values (10494, 'commi', 4, '1997-04-02', '1997-04-30', '1997-04-09', 2, 65.9899979, 'comércio mineiro', 'av. dos lusíadas, 23', 'sao paulo', 'sp', '05432-043', 'brazil');
insert into orders values (10495, 'laugb', 3, '1997-04-03', '1997-05-01', '1997-04-11', 3, 4.6500001, 'laughing bacchus wine cellars', '2319 elm st.', 'vancouver', 'bc', 'v3f 2k1', 'canada');
insert into orders values (10496, 'tradh', 7, '1997-04-04', '1997-05-02', '1997-04-07', 2, 46.7700005, 'tradiçao hipermercados', 'av. inês de castro, 414', 'sao paulo', 'sp', '05634-030', 'brazil');
insert into orders values (10497, 'lehms', 7, '1997-04-04', '1997-05-02', '1997-04-07', 1, 36.2099991, 'lehmanns marktstand', 'magazinweg 7', 'frankfurt a.m.', null, '60528', 'germany');
insert into orders values (10498, 'hilaa', 8, '1997-04-07', '1997-05-05', '1997-04-11', 2, 29.75, 'hilarion-abastos', 'carrera 22 con ave. carlos soublette #8-35', 'san cristóbal', 'táchira', '5022', 'venezuela');
insert into orders values (10499, 'lilas', 4, '1997-04-08', '1997-05-06', '1997-04-16', 2, 102.019997, 'lila-supermercado', 'carrera 52 con ave. bolívar #65-98 llano largo', 'barquisimeto', 'lara', '3508', 'venezuela');
insert into orders values (10500, 'lamai', 6, '1997-04-09', '1997-05-07', '1997-04-17', 1, 42.6800003, 'la maison d''asie', '1 rue alsace-lorraine', 'toulouse', null, '31000', 'france');
insert into orders values (10501, 'blaus', 9, '1997-04-09', '1997-05-07', '1997-04-16', 3, 8.85000038, 'blauer see delikatessen', 'forsterstr. 57', 'mannheim', null, '68306', 'germany');
insert into orders values (10502, 'peric', 2, '1997-04-10', '1997-05-08', '1997-04-29', 1, 69.3199997, 'pericles comidas clásicas', 'calle dr. jorge cash 321', 'méxico d.f.', null, '05033', 'mexico');
insert into orders values (10503, 'hungo', 6, '1997-04-11', '1997-05-09', '1997-04-16', 2, 16.7399998, 'hungry owl all-night grocers', '8 johnstown road', 'cork', 'co. cork', null, 'ireland');
insert into orders values (10504, 'whitc', 4, '1997-04-11', '1997-05-09', '1997-04-18', 3, 59.1300011, 'white clover markets', '1029 - 12th ave. s.', 'seattle', 'wa', '98124', 'usa');
insert into orders values (10505, 'merep', 3, '1997-04-14', '1997-05-12', '1997-04-21', 3, 7.13000011, 'mère paillarde', '43 rue st. laurent', 'montréal', 'québec', 'h1j 1c3', 'canada');
insert into orders values (10506, 'koene', 9, '1997-04-15', '1997-05-13', '1997-05-02', 2, 21.1900005, 'königlich essen', 'maubelstr. 90', 'brandenburg', null, '14776', 'germany');
insert into orders values (10507, 'anton', 7, '1997-04-15', '1997-05-13', '1997-04-22', 1, 47.4500008, 'antonio moreno taquería', 'mataderos  2312', 'méxico d.f.', null, '05023', 'mexico');
insert into orders values (10508, 'ottik', 1, '1997-04-16', '1997-05-14', '1997-05-13', 2, 4.98999977, 'ottilies käseladen', 'mehrheimerstr. 369', 'köln', null, '50739', 'germany');
insert into orders values (10509, 'blaus', 4, '1997-04-17', '1997-05-15', '1997-04-29', 1, 0.150000006, 'blauer see delikatessen', 'forsterstr. 57', 'mannheim', null, '68306', 'germany');
insert into orders values (10510, 'savea', 6, '1997-04-18', '1997-05-16', '1997-04-28', 3, 367.630005, 'save-a-lot markets', '187 suffolk ln.', 'boise', 'id', '83720', 'usa');
insert into orders values (10511, 'bonap', 4, '1997-04-18', '1997-05-16', '1997-04-21', 3, 350.640015, 'bon app''', '12, rue des bouchers', 'marseille', null, '13008', 'france');
insert into orders values (10512, 'famia', 7, '1997-04-21', '1997-05-19', '1997-04-24', 2, 3.52999997, 'familia arquibaldo', 'rua orós, 92', 'sao paulo', 'sp', '05442-030', 'brazil');
insert into orders values (10513, 'wandk', 7, '1997-04-22', '1997-06-03', '1997-04-28', 1, 105.650002, 'die wandernde kuh', 'adenauerallee 900', 'stuttgart', null, '70563', 'germany');
insert into orders values (10514, 'ernsh', 3, '1997-04-22', '1997-05-20', '1997-05-16', 2, 789.950012, 'ernst handel', 'kirchgasse 6', 'graz', null, '8010', 'austria');
insert into orders values (10515, 'quick', 2, '1997-04-23', '1997-05-07', '1997-05-23', 1, 204.470001, 'quick-stop', 'taucherstraße 10', 'cunewalde', null, '01307', 'germany');
insert into orders values (10516, 'hungo', 2, '1997-04-24', '1997-05-22', '1997-05-01', 3, 62.7799988, 'hungry owl all-night grocers', '8 johnstown road', 'cork', 'co. cork', null, 'ireland');
insert into orders values (10517, 'norts', 3, '1997-04-24', '1997-05-22', '1997-04-29', 3, 32.0699997, 'north/south', 'south house 300 queensbridge', 'london', null, 'sw7 1rz', 'uk');
insert into orders values (10518, 'tortu', 4, '1997-04-25', '1997-05-09', '1997-05-05', 2, 218.149994, 'tortuga restaurante', 'avda. azteca 123', 'méxico d.f.', null, '05033', 'mexico');
insert into orders values (10519, 'chops', 6, '1997-04-28', '1997-05-26', '1997-05-01', 3, 91.7600021, 'chop-suey chinese', 'hauptstr. 31', 'bern', null, '3012', 'switzerland');
insert into orders values (10520, 'santg', 7, '1997-04-29', '1997-05-27', '1997-05-01', 1, 13.3699999, 'santé gourmet', 'erling skakkes gate 78', 'stavern', null, '4110', 'norway');
insert into orders values (10521, 'cactu', 8, '1997-04-29', '1997-05-27', '1997-05-02', 2, 17.2199993, 'cactus comidas para llevar', 'cerrito 333', 'buenos aires', null, '1010', 'argentina');
insert into orders values (10522, 'lehms', 4, '1997-04-30', '1997-05-28', '1997-05-06', 1, 45.3300018, 'lehmanns marktstand', 'magazinweg 7', 'frankfurt a.m.', null, '60528', 'germany');
insert into orders values (10523, 'seves', 7, '1997-05-01', '1997-05-29', '1997-05-30', 2, 77.6299973, 'seven seas imports', '90 wadhurst rd.', 'london', null, 'ox15 4nb', 'uk');
insert into orders values (10524, 'bergs', 1, '1997-05-01', '1997-05-29', '1997-05-07', 2, 244.789993, 'berglunds snabbköp', 'berguvsvägen  8', 'luleå', null, 's-958 22', 'sweden');
insert into orders values (10525, 'bonap', 1, '1997-05-02', '1997-05-30', '1997-05-23', 2, 11.0600004, 'bon app''', '12, rue des bouchers', 'marseille', null, '13008', 'france');
insert into orders values (10526, 'warth', 4, '1997-05-05', '1997-06-02', '1997-05-15', 2, 58.5900002, 'wartian herkku', 'torikatu 38', 'oulu', null, '90110', 'finland');
insert into orders values (10527, 'quick', 7, '1997-05-05', '1997-06-02', '1997-05-07', 1, 41.9000015, 'quick-stop', 'taucherstraße 10', 'cunewalde', null, '01307', 'germany');
insert into orders values (10528, 'greal', 6, '1997-05-06', '1997-05-20', '1997-05-09', 2, 3.3499999, 'great lakes food market', '2732 baker blvd.', 'eugene', 'or', '97403', 'usa');
insert into orders values (10529, 'maisd', 5, '1997-05-07', '1997-06-04', '1997-05-09', 2, 66.6900024, 'maison dewey', 'rue joseph-bens 532', 'bruxelles', null, 'b-1180', 'belgium');
insert into orders values (10530, 'picco', 3, '1997-05-08', '1997-06-05', '1997-05-12', 2, 339.220001, 'piccolo und mehr', 'geislweg 14', 'salzburg', null, '5020', 'austria');
insert into orders values (10531, 'ocean', 7, '1997-05-08', '1997-06-05', '1997-05-19', 1, 8.11999989, 'océano atlántico ltda.', 'ing. gustavo moncada 8585 piso 20-a', 'buenos aires', null, '1010', 'argentina');
insert into orders values (10532, 'eastc', 7, '1997-05-09', '1997-06-06', '1997-05-12', 3, 74.4599991, 'eastern connection', '35 king george', 'london', null, 'wx3 6fw', 'uk');
insert into orders values (10533, 'folko', 8, '1997-05-12', '1997-06-09', '1997-05-22', 1, 188.039993, 'folk och fä hb', 'åkergatan 24', 'bräcke', null, 's-844 67', 'sweden');
insert into orders values (10534, 'lehms', 8, '1997-05-12', '1997-06-09', '1997-05-14', 2, 27.9400005, 'lehmanns marktstand', 'magazinweg 7', 'frankfurt a.m.', null, '60528', 'germany');
insert into orders values (10535, 'anton', 4, '1997-05-13', '1997-06-10', '1997-05-21', 1, 15.6400003, 'antonio moreno taquería', 'mataderos  2312', 'méxico d.f.', null, '05023', 'mexico');
insert into orders values (10536, 'lehms', 3, '1997-05-14', '1997-06-11', '1997-06-06', 2, 58.8800011, 'lehmanns marktstand', 'magazinweg 7', 'frankfurt a.m.', null, '60528', 'germany');
insert into orders values (10537, 'ricsu', 1, '1997-05-14', '1997-05-28', '1997-05-19', 1, 78.8499985, 'richter supermarkt', 'starenweg 5', 'genève', null, '1204', 'switzerland');
insert into orders values (10538, 'bsbev', 9, '1997-05-15', '1997-06-12', '1997-05-16', 3, 4.86999989, 'b''s beverages', 'fauntleroy circus', 'london', null, 'ec2 5nt', 'uk');
insert into orders values (10539, 'bsbev', 6, '1997-05-16', '1997-06-13', '1997-05-23', 3, 12.3599997, 'b''s beverages', 'fauntleroy circus', 'london', null, 'ec2 5nt', 'uk');
insert into orders values (10540, 'quick', 3, '1997-05-19', '1997-06-16', '1997-06-13', 3, 1007.64001, 'quick-stop', 'taucherstraße 10', 'cunewalde', null, '01307', 'germany');
insert into orders values (10541, 'hanar', 2, '1997-05-19', '1997-06-16', '1997-05-29', 1, 68.6500015, 'hanari carnes', 'rua do paço, 67', 'rio de janeiro', 'rj', '05454-876', 'brazil');
insert into orders values (10542, 'koene', 1, '1997-05-20', '1997-06-17', '1997-05-26', 3, 10.9499998, 'königlich essen', 'maubelstr. 90', 'brandenburg', null, '14776', 'germany');
insert into orders values (10543, 'lilas', 8, '1997-05-21', '1997-06-18', '1997-05-23', 2, 48.1699982, 'lila-supermercado', 'carrera 52 con ave. bolívar #65-98 llano largo', 'barquisimeto', 'lara', '3508', 'venezuela');
insert into orders values (10544, 'lonep', 4, '1997-05-21', '1997-06-18', '1997-05-30', 1, 24.9099998, 'lonesome pine restaurant', '89 chiaroscuro rd.', 'portland', 'or', '97219', 'usa');
insert into orders values (10545, 'lazyk', 8, '1997-05-22', '1997-06-19', '1997-06-26', 2, 11.9200001, 'lazy k kountry store', '12 orchestra terrace', 'walla walla', 'wa', '99362', 'usa');
insert into orders values (10546, 'victe', 1, '1997-05-23', '1997-06-20', '1997-05-27', 3, 194.720001, 'victuailles en stock', '2, rue du commerce', 'lyon', null, '69004', 'france');
insert into orders values (10547, 'seves', 3, '1997-05-23', '1997-06-20', '1997-06-02', 2, 178.429993, 'seven seas imports', '90 wadhurst rd.', 'london', null, 'ox15 4nb', 'uk');
insert into orders values (10548, 'tomsp', 3, '1997-05-26', '1997-06-23', '1997-06-02', 2, 1.42999995, 'toms spezialitäten', 'luisenstr. 48', 'münster', null, '44087', 'germany');
insert into orders values (10549, 'quick', 5, '1997-05-27', '1997-06-10', '1997-05-30', 1, 171.240005, 'quick-stop', 'taucherstraße 10', 'cunewalde', null, '01307', 'germany');
insert into orders values (10550, 'godos', 7, '1997-05-28', '1997-06-25', '1997-06-06', 3, 4.32000017, 'godos cocina típica', 'c/ romero, 33', 'sevilla', null, '41101', 'spain');
insert into orders values (10551, 'furib', 4, '1997-05-28', '1997-07-09', '1997-06-06', 3, 72.9499969, 'furia bacalhau e frutos do mar', 'jardim das rosas n. 32', 'lisboa', null, '1675', 'portugal');
insert into orders values (10552, 'hilaa', 2, '1997-05-29', '1997-06-26', '1997-06-05', 1, 83.2200012, 'hilarion-abastos', 'carrera 22 con ave. carlos soublette #8-35', 'san cristóbal', 'táchira', '5022', 'venezuela');
insert into orders values (10553, 'warth', 2, '1997-05-30', '1997-06-27', '1997-06-03', 2, 149.490005, 'wartian herkku', 'torikatu 38', 'oulu', null, '90110', 'finland');
insert into orders values (10554, 'ottik', 4, '1997-05-30', '1997-06-27', '1997-06-05', 3, 120.970001, 'ottilies käseladen', 'mehrheimerstr. 369', 'köln', null, '50739', 'germany');
insert into orders values (10555, 'savea', 6, '1997-06-02', '1997-06-30', '1997-06-04', 3, 252.490005, 'save-a-lot markets', '187 suffolk ln.', 'boise', 'id', '83720', 'usa');
insert into orders values (10556, 'simob', 2, '1997-06-03', '1997-07-15', '1997-06-13', 1, 9.80000019, 'simons bistro', 'vinbæltet 34', 'kobenhavn', null, '1734', 'denmark');
insert into orders values (10557, 'lehms', 9, '1997-06-03', '1997-06-17', '1997-06-06', 2, 96.7200012, 'lehmanns marktstand', 'magazinweg 7', 'frankfurt a.m.', null, '60528', 'germany');
insert into orders values (10558, 'arout', 1, '1997-06-04', '1997-07-02', '1997-06-10', 2, 72.9700012, 'around the horn', 'brook farm stratford st. mary', 'colchester', 'essex', 'co7 6jx', 'uk');
insert into orders values (10559, 'blonp', 6, '1997-06-05', '1997-07-03', '1997-06-13', 1, 8.05000019, 'blondel père et fils', '24, place kléber', 'strasbourg', null, '67000', 'france');
insert into orders values (10560, 'frank', 8, '1997-06-06', '1997-07-04', '1997-06-09', 1, 36.6500015, 'frankenversand', 'berliner platz 43', 'münchen', null, '80805', 'germany');
insert into orders values (10561, 'folko', 2, '1997-06-06', '1997-07-04', '1997-06-09', 2, 242.210007, 'folk och fä hb', 'åkergatan 24', 'bräcke', null, 's-844 67', 'sweden');
insert into orders values (10562, 'reggc', 1, '1997-06-09', '1997-07-07', '1997-06-12', 1, 22.9500008, 'reggiani caseifici', 'strada provinciale 124', 'reggio emilia', null, '42100', 'italy');
insert into orders values (10563, 'ricar', 2, '1997-06-10', '1997-07-22', '1997-06-24', 2, 60.4300003, 'ricardo adocicados', 'av. copacabana, 267', 'rio de janeiro', 'rj', '02389-890', 'brazil');
insert into orders values (10564, 'rattc', 4, '1997-06-10', '1997-07-08', '1997-06-16', 3, 13.75, 'rattlesnake canyon grocery', '2817 milton dr.', 'albuquerque', 'nm', '87110', 'usa');
insert into orders values (10565, 'merep', 8, '1997-06-11', '1997-07-09', '1997-06-18', 2, 7.1500001, 'mère paillarde', '43 rue st. laurent', 'montréal', 'québec', 'h1j 1c3', 'canada');
insert into orders values (10566, 'blonp', 9, '1997-06-12', '1997-07-10', '1997-06-18', 1, 88.4000015, 'blondel père et fils', '24, place kléber', 'strasbourg', null, '67000', 'france');
insert into orders values (10567, 'hungo', 1, '1997-06-12', '1997-07-10', '1997-06-17', 1, 33.9700012, 'hungry owl all-night grocers', '8 johnstown road', 'cork', 'co. cork', null, 'ireland');
insert into orders values (10568, 'galed', 3, '1997-06-13', '1997-07-11', '1997-07-09', 3, 6.53999996, 'galería del gastronómo', 'rambla de cataluña, 23', 'barcelona', null, '8022', 'spain');
insert into orders values (10569, 'rattc', 5, '1997-06-16', '1997-07-14', '1997-07-11', 1, 58.9799995, 'rattlesnake canyon grocery', '2817 milton dr.', 'albuquerque', 'nm', '87110', 'usa');
insert into orders values (10570, 'merep', 3, '1997-06-17', '1997-07-15', '1997-06-19', 3, 188.990005, 'mère paillarde', '43 rue st. laurent', 'montréal', 'québec', 'h1j 1c3', 'canada');
insert into orders values (10571, 'ernsh', 8, '1997-06-17', '1997-07-29', '1997-07-04', 3, 26.0599995, 'ernst handel', 'kirchgasse 6', 'graz', null, '8010', 'austria');
insert into orders values (10572, 'bergs', 3, '1997-06-18', '1997-07-16', '1997-06-25', 2, 116.43, 'berglunds snabbköp', 'berguvsvägen  8', 'luleå', null, 's-958 22', 'sweden');
insert into orders values (10573, 'anton', 7, '1997-06-19', '1997-07-17', '1997-06-20', 3, 84.8399963, 'antonio moreno taquería', 'mataderos  2312', 'méxico d.f.', null, '05023', 'mexico');
insert into orders values (10574, 'traih', 4, '1997-06-19', '1997-07-17', '1997-06-30', 2, 37.5999985, 'trail''s head gourmet provisioners', '722 davinci blvd.', 'kirkland', 'wa', '98034', 'usa');
insert into orders values (10575, 'morgk', 5, '1997-06-20', '1997-07-04', '1997-06-30', 1, 127.339996, 'morgenstern gesundkost', 'heerstr. 22', 'leipzig', null, '04179', 'germany');
insert into orders values (10576, 'tortu', 3, '1997-06-23', '1997-07-07', '1997-06-30', 3, 18.5599995, 'tortuga restaurante', 'avda. azteca 123', 'méxico d.f.', null, '05033', 'mexico');
insert into orders values (10577, 'traih', 9, '1997-06-23', '1997-08-04', '1997-06-30', 2, 25.4099998, 'trail''s head gourmet provisioners', '722 davinci blvd.', 'kirkland', 'wa', '98034', 'usa');
insert into orders values (10578, 'bsbev', 4, '1997-06-24', '1997-07-22', '1997-07-25', 3, 29.6000004, 'b''s beverages', 'fauntleroy circus', 'london', null, 'ec2 5nt', 'uk');
insert into orders values (10579, 'letss', 1, '1997-06-25', '1997-07-23', '1997-07-04', 2, 13.7299995, 'let''s stop n shop', '87 polk st. suite 5', 'san francisco', 'ca', '94117', 'usa');
insert into orders values (10580, 'ottik', 4, '1997-06-26', '1997-07-24', '1997-07-01', 3, 75.8899994, 'ottilies käseladen', 'mehrheimerstr. 369', 'köln', null, '50739', 'germany');
insert into orders values (10581, 'famia', 3, '1997-06-26', '1997-07-24', '1997-07-02', 1, 3.00999999, 'familia arquibaldo', 'rua orós, 92', 'sao paulo', 'sp', '05442-030', 'brazil');
insert into orders values (10582, 'blaus', 3, '1997-06-27', '1997-07-25', '1997-07-14', 2, 27.7099991, 'blauer see delikatessen', 'forsterstr. 57', 'mannheim', null, '68306', 'germany');
insert into orders values (10583, 'warth', 2, '1997-06-30', '1997-07-28', '1997-07-04', 2, 7.28000021, 'wartian herkku', 'torikatu 38', 'oulu', null, '90110', 'finland');
insert into orders values (10584, 'blonp', 4, '1997-06-30', '1997-07-28', '1997-07-04', 1, 59.1399994, 'blondel père et fils', '24, place kléber', 'strasbourg', null, '67000', 'france');
insert into orders values (10585, 'welli', 7, '1997-07-01', '1997-07-29', '1997-07-10', 1, 13.4099998, 'wellington importadora', 'rua do mercado, 12', 'resende', 'sp', '08737-363', 'brazil');
insert into orders values (10586, 'reggc', 9, '1997-07-02', '1997-07-30', '1997-07-09', 1, 0.479999989, 'reggiani caseifici', 'strada provinciale 124', 'reggio emilia', null, '42100', 'italy');
insert into orders values (10587, 'quede', 1, '1997-07-02', '1997-07-30', '1997-07-09', 1, 62.5200005, 'que delícia', 'rua da panificadora, 12', 'rio de janeiro', 'rj', '02389-673', 'brazil');
insert into orders values (10588, 'quick', 2, '1997-07-03', '1997-07-31', '1997-07-10', 3, 194.669998, 'quick-stop', 'taucherstraße 10', 'cunewalde', null, '01307', 'germany');
insert into orders values (10589, 'greal', 8, '1997-07-04', '1997-08-01', '1997-07-14', 2, 4.42000008, 'great lakes food market', '2732 baker blvd.', 'eugene', 'or', '97403', 'usa');
insert into orders values (10590, 'merep', 4, '1997-07-07', '1997-08-04', '1997-07-14', 3, 44.7700005, 'mère paillarde', '43 rue st. laurent', 'montréal', 'québec', 'h1j 1c3', 'canada');
insert into orders values (10591, 'vaffe', 1, '1997-07-07', '1997-07-21', '1997-07-16', 1, 55.9199982, 'vaffeljernet', 'smagsloget 45', 'århus', null, '8200', 'denmark');
insert into orders values (10592, 'lehms', 3, '1997-07-08', '1997-08-05', '1997-07-16', 1, 32.0999985, 'lehmanns marktstand', 'magazinweg 7', 'frankfurt a.m.', null, '60528', 'germany');
insert into orders values (10593, 'lehms', 7, '1997-07-09', '1997-08-06', '1997-08-13', 2, 174.199997, 'lehmanns marktstand', 'magazinweg 7', 'frankfurt a.m.', null, '60528', 'germany');
insert into orders values (10594, 'oldwo', 3, '1997-07-09', '1997-08-06', '1997-07-16', 2, 5.23999977, 'old world delicatessen', '2743 bering st.', 'anchorage', 'ak', '99508', 'usa');
insert into orders values (10595, 'ernsh', 2, '1997-07-10', '1997-08-07', '1997-07-14', 1, 96.7799988, 'ernst handel', 'kirchgasse 6', 'graz', null, '8010', 'austria');
insert into orders values (10596, 'whitc', 8, '1997-07-11', '1997-08-08', '1997-08-12', 1, 16.3400002, 'white clover markets', '1029 - 12th ave. s.', 'seattle', 'wa', '98124', 'usa');
insert into orders values (10597, 'picco', 7, '1997-07-11', '1997-08-08', '1997-07-18', 3, 35.1199989, 'piccolo und mehr', 'geislweg 14', 'salzburg', null, '5020', 'austria');
insert into orders values (10598, 'rattc', 1, '1997-07-14', '1997-08-11', '1997-07-18', 3, 44.4199982, 'rattlesnake canyon grocery', '2817 milton dr.', 'albuquerque', 'nm', '87110', 'usa');
insert into orders values (10599, 'bsbev', 6, '1997-07-15', '1997-08-26', '1997-07-21', 3, 29.9799995, 'b''s beverages', 'fauntleroy circus', 'london', null, 'ec2 5nt', 'uk');
insert into orders values (10600, 'hungc', 4, '1997-07-16', '1997-08-13', '1997-07-21', 1, 45.1300011, 'hungry coyote import store', 'city center plaza 516 main st.', 'elgin', 'or', '97827', 'usa');
insert into orders values (10601, 'hilaa', 7, '1997-07-16', '1997-08-27', '1997-07-22', 1, 58.2999992, 'hilarion-abastos', 'carrera 22 con ave. carlos soublette #8-35', 'san cristóbal', 'táchira', '5022', 'venezuela');
insert into orders values (10602, 'vaffe', 8, '1997-07-17', '1997-08-14', '1997-07-22', 2, 2.92000008, 'vaffeljernet', 'smagsloget 45', 'århus', null, '8200', 'denmark');
insert into orders values (10603, 'savea', 8, '1997-07-18', '1997-08-15', '1997-08-08', 2, 48.7700005, 'save-a-lot markets', '187 suffolk ln.', 'boise', 'id', '83720', 'usa');
insert into orders values (10604, 'furib', 1, '1997-07-18', '1997-08-15', '1997-07-29', 1, 7.46000004, 'furia bacalhau e frutos do mar', 'jardim das rosas n. 32', 'lisboa', null, '1675', 'portugal');
insert into orders values (10605, 'merep', 1, '1997-07-21', '1997-08-18', '1997-07-29', 2, 379.130005, 'mère paillarde', '43 rue st. laurent', 'montréal', 'québec', 'h1j 1c3', 'canada');
insert into orders values (10606, 'tradh', 4, '1997-07-22', '1997-08-19', '1997-07-31', 3, 79.4000015, 'tradiçao hipermercados', 'av. inês de castro, 414', 'sao paulo', 'sp', '05634-030', 'brazil');
insert into orders values (10607, 'savea', 5, '1997-07-22', '1997-08-19', '1997-07-25', 1, 200.240005, 'save-a-lot markets', '187 suffolk ln.', 'boise', 'id', '83720', 'usa');
insert into orders values (10608, 'tomsp', 4, '1997-07-23', '1997-08-20', '1997-08-01', 2, 27.7900009, 'toms spezialitäten', 'luisenstr. 48', 'münster', null, '44087', 'germany');
insert into orders values (10609, 'dumon', 7, '1997-07-24', '1997-08-21', '1997-07-30', 2, 1.85000002, 'du monde entier', '67, rue des cinquante otages', 'nantes', null, '44000', 'france');
insert into orders values (10610, 'lamai', 8, '1997-07-25', '1997-08-22', '1997-08-06', 1, 26.7800007, 'la maison d''asie', '1 rue alsace-lorraine', 'toulouse', null, '31000', 'france');
insert into orders values (10611, 'wolza', 6, '1997-07-25', '1997-08-22', '1997-08-01', 2, 80.6500015, 'wolski zajazd', 'ul. filtrowa 68', 'warszawa', null, '01-012', 'poland');
insert into orders values (10612, 'savea', 1, '1997-07-28', '1997-08-25', '1997-08-01', 2, 544.080017, 'save-a-lot markets', '187 suffolk ln.', 'boise', 'id', '83720', 'usa');
insert into orders values (10613, 'hilaa', 4, '1997-07-29', '1997-08-26', '1997-08-01', 2, 8.10999966, 'hilarion-abastos', 'carrera 22 con ave. carlos soublette #8-35', 'san cristóbal', 'táchira', '5022', 'venezuela');
insert into orders values (10614, 'blaus', 8, '1997-07-29', '1997-08-26', '1997-08-01', 3, 1.92999995, 'blauer see delikatessen', 'forsterstr. 57', 'mannheim', null, '68306', 'germany');
insert into orders values (10615, 'wilmk', 2, '1997-07-30', '1997-08-27', '1997-08-06', 3, 0.75, 'wilman kala', 'keskuskatu 45', 'helsinki', null, '21240', 'finland');
insert into orders values (10616, 'greal', 1, '1997-07-31', '1997-08-28', '1997-08-05', 2, 116.529999, 'great lakes food market', '2732 baker blvd.', 'eugene', 'or', '97403', 'usa');
insert into orders values (10617, 'greal', 4, '1997-07-31', '1997-08-28', '1997-08-04', 2, 18.5300007, 'great lakes food market', '2732 baker blvd.', 'eugene', 'or', '97403', 'usa');
insert into orders values (10618, 'merep', 1, '1997-08-01', '1997-09-12', '1997-08-08', 1, 154.679993, 'mère paillarde', '43 rue st. laurent', 'montréal', 'québec', 'h1j 1c3', 'canada');
insert into orders values (10619, 'merep', 3, '1997-08-04', '1997-09-01', '1997-08-07', 3, 91.0500031, 'mère paillarde', '43 rue st. laurent', 'montréal', 'québec', 'h1j 1c3', 'canada');
insert into orders values (10620, 'laugb', 2, '1997-08-05', '1997-09-02', '1997-08-14', 3, 0.939999998, 'laughing bacchus wine cellars', '2319 elm st.', 'vancouver', 'bc', 'v3f 2k1', 'canada');
insert into orders values (10621, 'islat', 4, '1997-08-05', '1997-09-02', '1997-08-11', 2, 23.7299995, 'island trading', 'garden house crowther way', 'cowes', 'isle of wight', 'po31 7pj', 'uk');
insert into orders values (10622, 'ricar', 4, '1997-08-06', '1997-09-03', '1997-08-11', 3, 50.9700012, 'ricardo adocicados', 'av. copacabana, 267', 'rio de janeiro', 'rj', '02389-890', 'brazil');
insert into orders values (10623, 'frank', 8, '1997-08-07', '1997-09-04', '1997-08-12', 2, 97.1800003, 'frankenversand', 'berliner platz 43', 'münchen', null, '80805', 'germany');
insert into orders values (10624, 'thecr', 4, '1997-08-07', '1997-09-04', '1997-08-19', 2, 94.8000031, 'the cracker box', '55 grizzly peak rd.', 'butte', 'mt', '59801', 'usa');
insert into orders values (10625, 'anatr', 3, '1997-08-08', '1997-09-05', '1997-08-14', 1, 43.9000015, 'ana trujillo emparedados y helados', 'avda. de la constitución 2222', 'méxico d.f.', null, '05021', 'mexico');
insert into orders values (10626, 'bergs', 1, '1997-08-11', '1997-09-08', '1997-08-20', 2, 138.690002, 'berglunds snabbköp', 'berguvsvägen  8', 'luleå', null, 's-958 22', 'sweden');
insert into orders values (10627, 'savea', 8, '1997-08-11', '1997-09-22', '1997-08-21', 3, 107.459999, 'save-a-lot markets', '187 suffolk ln.', 'boise', 'id', '83720', 'usa');
insert into orders values (10628, 'blonp', 4, '1997-08-12', '1997-09-09', '1997-08-20', 3, 30.3600006, 'blondel père et fils', '24, place kléber', 'strasbourg', null, '67000', 'france');
insert into orders values (10629, 'godos', 4, '1997-08-12', '1997-09-09', '1997-08-20', 3, 85.4599991, 'godos cocina típica', 'c/ romero, 33', 'sevilla', null, '41101', 'spain');
insert into orders values (10630, 'koene', 1, '1997-08-13', '1997-09-10', '1997-08-19', 2, 32.3499985, 'königlich essen', 'maubelstr. 90', 'brandenburg', null, '14776', 'germany');
insert into orders values (10631, 'lamai', 8, '1997-08-14', '1997-09-11', '1997-08-15', 1, 0.870000005, 'la maison d''asie', '1 rue alsace-lorraine', 'toulouse', null, '31000', 'france');
insert into orders values (10632, 'wandk', 8, '1997-08-14', '1997-09-11', '1997-08-19', 1, 41.3800011, 'die wandernde kuh', 'adenauerallee 900', 'stuttgart', null, '70563', 'germany');
insert into orders values (10633, 'ernsh', 7, '1997-08-15', '1997-09-12', '1997-08-18', 3, 477.899994, 'ernst handel', 'kirchgasse 6', 'graz', null, '8010', 'austria');
insert into orders values (10634, 'folig', 4, '1997-08-15', '1997-09-12', '1997-08-21', 3, 487.380005, 'folies gourmandes', '184, chaussée de tournai', 'lille', null, '59000', 'france');
insert into orders values (10635, 'magaa', 8, '1997-08-18', '1997-09-15', '1997-08-21', 3, 47.4599991, 'magazzini alimentari riuniti', 'via ludovico il moro 22', 'bergamo', null, '24100', 'italy');
insert into orders values (10636, 'warth', 4, '1997-08-19', '1997-09-16', '1997-08-26', 1, 1.14999998, 'wartian herkku', 'torikatu 38', 'oulu', null, '90110', 'finland');
insert into orders values (10637, 'queen', 6, '1997-08-19', '1997-09-16', '1997-08-26', 1, 201.289993, 'queen cozinha', 'alameda dos canàrios, 891', 'sao paulo', 'sp', '05487-020', 'brazil');
insert into orders values (10638, 'linod', 3, '1997-08-20', '1997-09-17', '1997-09-01', 1, 158.440002, 'lino-delicateses', 'ave. 5 de mayo porlamar', 'i. de margarita', 'nueva esparta', '4980', 'venezuela');
insert into orders values (10639, 'santg', 7, '1997-08-20', '1997-09-17', '1997-08-27', 3, 38.6399994, 'santé gourmet', 'erling skakkes gate 78', 'stavern', null, '4110', 'norway');
insert into orders values (10640, 'wandk', 4, '1997-08-21', '1997-09-18', '1997-08-28', 1, 23.5499992, 'die wandernde kuh', 'adenauerallee 900', 'stuttgart', null, '70563', 'germany');
insert into orders values (10641, 'hilaa', 4, '1997-08-22', '1997-09-19', '1997-08-26', 2, 179.610001, 'hilarion-abastos', 'carrera 22 con ave. carlos soublette #8-35', 'san cristóbal', 'táchira', '5022', 'venezuela');
insert into orders values (10642, 'simob', 7, '1997-08-22', '1997-09-19', '1997-09-05', 3, 41.8899994, 'simons bistro', 'vinbæltet 34', 'kobenhavn', null, '1734', 'denmark');
insert into orders values (10643, 'alfki', 6, '1997-08-25', '1997-09-22', '1997-09-02', 1, 29.4599991, 'alfreds futterkiste', 'obere str. 57', 'berlin', null, '12209', 'germany');
insert into orders values (10644, 'welli', 3, '1997-08-25', '1997-09-22', '1997-09-01', 2, 0.140000001, 'wellington importadora', 'rua do mercado, 12', 'resende', 'sp', '08737-363', 'brazil');
insert into orders values (10645, 'hanar', 4, '1997-08-26', '1997-09-23', '1997-09-02', 1, 12.4099998, 'hanari carnes', 'rua do paço, 67', 'rio de janeiro', 'rj', '05454-876', 'brazil');
insert into orders values (10646, 'hungo', 9, '1997-08-27', '1997-10-08', '1997-09-03', 3, 142.330002, 'hungry owl all-night grocers', '8 johnstown road', 'cork', 'co. cork', null, 'ireland');
insert into orders values (10647, 'quede', 4, '1997-08-27', '1997-09-10', '1997-09-03', 2, 45.5400009, 'que delícia', 'rua da panificadora, 12', 'rio de janeiro', 'rj', '02389-673', 'brazil');
insert into orders values (10648, 'ricar', 5, '1997-08-28', '1997-10-09', '1997-09-09', 2, 14.25, 'ricardo adocicados', 'av. copacabana, 267', 'rio de janeiro', 'rj', '02389-890', 'brazil');
insert into orders values (10649, 'maisd', 5, '1997-08-28', '1997-09-25', '1997-08-29', 3, 6.19999981, 'maison dewey', 'rue joseph-bens 532', 'bruxelles', null, 'b-1180', 'belgium');
insert into orders values (10650, 'famia', 5, '1997-08-29', '1997-09-26', '1997-09-03', 3, 176.809998, 'familia arquibaldo', 'rua orós, 92', 'sao paulo', 'sp', '05442-030', 'brazil');
insert into orders values (10651, 'wandk', 8, '1997-09-01', '1997-09-29', '1997-09-11', 2, 20.6000004, 'die wandernde kuh', 'adenauerallee 900', 'stuttgart', null, '70563', 'germany');
insert into orders values (10652, 'gourl', 4, '1997-09-01', '1997-09-29', '1997-09-08', 2, 7.13999987, 'gourmet lanchonetes', 'av. brasil, 442', 'campinas', 'sp', '04876-786', 'brazil');
insert into orders values (10653, 'frank', 1, '1997-09-02', '1997-09-30', '1997-09-19', 1, 93.25, 'frankenversand', 'berliner platz 43', 'münchen', null, '80805', 'germany');
insert into orders values (10654, 'bergs', 5, '1997-09-02', '1997-09-30', '1997-09-11', 1, 55.2599983, 'berglunds snabbköp', 'berguvsvägen  8', 'luleå', null, 's-958 22', 'sweden');
insert into orders values (10655, 'reggc', 1, '1997-09-03', '1997-10-01', '1997-09-11', 2, 4.40999985, 'reggiani caseifici', 'strada provinciale 124', 'reggio emilia', null, '42100', 'italy');
insert into orders values (10656, 'greal', 6, '1997-09-04', '1997-10-02', '1997-09-10', 1, 57.1500015, 'great lakes food market', '2732 baker blvd.', 'eugene', 'or', '97403', 'usa');
insert into orders values (10657, 'savea', 2, '1997-09-04', '1997-10-02', '1997-09-15', 2, 352.690002, 'save-a-lot markets', '187 suffolk ln.', 'boise', 'id', '83720', 'usa');
insert into orders values (10658, 'quick', 4, '1997-09-05', '1997-10-03', '1997-09-08', 1, 364.149994, 'quick-stop', 'taucherstraße 10', 'cunewalde', null, '01307', 'germany');
insert into orders values (10659, 'queen', 7, '1997-09-05', '1997-10-03', '1997-09-10', 2, 105.809998, 'queen cozinha', 'alameda dos canàrios, 891', 'sao paulo', 'sp', '05487-020', 'brazil');
insert into orders values (10660, 'hungc', 8, '1997-09-08', '1997-10-06', '1997-10-15', 1, 111.290001, 'hungry coyote import store', 'city center plaza 516 main st.', 'elgin', 'or', '97827', 'usa');
insert into orders values (10661, 'hungo', 7, '1997-09-09', '1997-10-07', '1997-09-15', 3, 17.5499992, 'hungry owl all-night grocers', '8 johnstown road', 'cork', 'co. cork', null, 'ireland');
insert into orders values (10662, 'lonep', 3, '1997-09-09', '1997-10-07', '1997-09-18', 2, 1.27999997, 'lonesome pine restaurant', '89 chiaroscuro rd.', 'portland', 'or', '97219', 'usa');
insert into orders values (10663, 'bonap', 2, '1997-09-10', '1997-09-24', '1997-10-03', 2, 113.150002, 'bon app''', '12, rue des bouchers', 'marseille', null, '13008', 'france');
insert into orders values (10664, 'furib', 1, '1997-09-10', '1997-10-08', '1997-09-19', 3, 1.26999998, 'furia bacalhau e frutos do mar', 'jardim das rosas n. 32', 'lisboa', null, '1675', 'portugal');
insert into orders values (10665, 'lonep', 1, '1997-09-11', '1997-10-09', '1997-09-17', 2, 26.3099995, 'lonesome pine restaurant', '89 chiaroscuro rd.', 'portland', 'or', '97219', 'usa');
insert into orders values (10666, 'ricsu', 7, '1997-09-12', '1997-10-10', '1997-09-22', 2, 232.419998, 'richter supermarkt', 'starenweg 5', 'genève', null, '1204', 'switzerland');
insert into orders values (10667, 'ernsh', 7, '1997-09-12', '1997-10-10', '1997-09-19', 1, 78.0899963, 'ernst handel', 'kirchgasse 6', 'graz', null, '8010', 'austria');
insert into orders values (10668, 'wandk', 1, '1997-09-15', '1997-10-13', '1997-09-23', 2, 47.2200012, 'die wandernde kuh', 'adenauerallee 900', 'stuttgart', null, '70563', 'germany');
insert into orders values (10669, 'simob', 2, '1997-09-15', '1997-10-13', '1997-09-22', 1, 24.3899994, 'simons bistro', 'vinbæltet 34', 'kobenhavn', null, '1734', 'denmark');
insert into orders values (10670, 'frank', 4, '1997-09-16', '1997-10-14', '1997-09-18', 1, 203.479996, 'frankenversand', 'berliner platz 43', 'münchen', null, '80805', 'germany');
insert into orders values (10671, 'franr', 1, '1997-09-17', '1997-10-15', '1997-09-24', 1, 30.3400002, 'france restauration', '54, rue royale', 'nantes', null, '44000', 'france');
insert into orders values (10672, 'bergs', 9, '1997-09-17', '1997-10-01', '1997-09-26', 2, 95.75, 'berglunds snabbköp', 'berguvsvägen  8', 'luleå', null, 's-958 22', 'sweden');
insert into orders values (10673, 'wilmk', 2, '1997-09-18', '1997-10-16', '1997-09-19', 1, 22.7600002, 'wilman kala', 'keskuskatu 45', 'helsinki', null, '21240', 'finland');
insert into orders values (10674, 'islat', 4, '1997-09-18', '1997-10-16', '1997-09-30', 2, 0.899999976, 'island trading', 'garden house crowther way', 'cowes', 'isle of wight', 'po31 7pj', 'uk');
insert into orders values (10675, 'frank', 5, '1997-09-19', '1997-10-17', '1997-09-23', 2, 31.8500004, 'frankenversand', 'berliner platz 43', 'münchen', null, '80805', 'germany');
insert into orders values (10676, 'tortu', 2, '1997-09-22', '1997-10-20', '1997-09-29', 2, 2.00999999, 'tortuga restaurante', 'avda. azteca 123', 'méxico d.f.', null, '05033', 'mexico');
insert into orders values (10677, 'anton', 1, '1997-09-22', '1997-10-20', '1997-09-26', 3, 4.03000021, 'antonio moreno taquería', 'mataderos  2312', 'méxico d.f.', null, '05023', 'mexico');
insert into orders values (10678, 'savea', 7, '1997-09-23', '1997-10-21', '1997-10-16', 3, 388.980011, 'save-a-lot markets', '187 suffolk ln.', 'boise', 'id', '83720', 'usa');
insert into orders values (10679, 'blonp', 8, '1997-09-23', '1997-10-21', '1997-09-30', 3, 27.9400005, 'blondel père et fils', '24, place kléber', 'strasbourg', null, '67000', 'france');
insert into orders values (10680, 'oldwo', 1, '1997-09-24', '1997-10-22', '1997-09-26', 1, 26.6100006, 'old world delicatessen', '2743 bering st.', 'anchorage', 'ak', '99508', 'usa');
insert into orders values (10681, 'greal', 3, '1997-09-25', '1997-10-23', '1997-09-30', 3, 76.1299973, 'great lakes food market', '2732 baker blvd.', 'eugene', 'or', '97403', 'usa');
insert into orders values (10682, 'anton', 3, '1997-09-25', '1997-10-23', '1997-10-01', 2, 36.1300011, 'antonio moreno taquería', 'mataderos  2312', 'méxico d.f.', null, '05023', 'mexico');
insert into orders values (10683, 'dumon', 2, '1997-09-26', '1997-10-24', '1997-10-01', 1, 4.4000001, 'du monde entier', '67, rue des cinquante otages', 'nantes', null, '44000', 'france');
insert into orders values (10684, 'ottik', 3, '1997-09-26', '1997-10-24', '1997-09-30', 1, 145.630005, 'ottilies käseladen', 'mehrheimerstr. 369', 'köln', null, '50739', 'germany');
insert into orders values (10685, 'gourl', 4, '1997-09-29', '1997-10-13', '1997-10-03', 2, 33.75, 'gourmet lanchonetes', 'av. brasil, 442', 'campinas', 'sp', '04876-786', 'brazil');
insert into orders values (10686, 'picco', 2, '1997-09-30', '1997-10-28', '1997-10-08', 1, 96.5, 'piccolo und mehr', 'geislweg 14', 'salzburg', null, '5020', 'austria');
insert into orders values (10687, 'hungo', 9, '1997-09-30', '1997-10-28', '1997-10-30', 2, 296.429993, 'hungry owl all-night grocers', '8 johnstown road', 'cork', 'co. cork', null, 'ireland');
insert into orders values (10688, 'vaffe', 4, '1997-10-01', '1997-10-15', '1997-10-07', 2, 299.089996, 'vaffeljernet', 'smagsloget 45', 'århus', null, '8200', 'denmark');
insert into orders values (10689, 'bergs', 1, '1997-10-01', '1997-10-29', '1997-10-07', 2, 13.4200001, 'berglunds snabbköp', 'berguvsvägen  8', 'luleå', null, 's-958 22', 'sweden');
insert into orders values (10690, 'hanar', 1, '1997-10-02', '1997-10-30', '1997-10-03', 1, 15.8000002, 'hanari carnes', 'rua do paço, 67', 'rio de janeiro', 'rj', '05454-876', 'brazil');
insert into orders values (10691, 'quick', 2, '1997-10-03', '1997-11-14', '1997-10-22', 2, 810.049988, 'quick-stop', 'taucherstraße 10', 'cunewalde', null, '01307', 'germany');
insert into orders values (10692, 'alfki', 4, '1997-10-03', '1997-10-31', '1997-10-13', 2, 61.0200005, 'alfred''s futterkiste', 'obere str. 57', 'berlin', null, '12209', 'germany');
insert into orders values (10693, 'whitc', 3, '1997-10-06', '1997-10-20', '1997-10-10', 3, 139.339996, 'white clover markets', '1029 - 12th ave. s.', 'seattle', 'wa', '98124', 'usa');
insert into orders values (10694, 'quick', 8, '1997-10-06', '1997-11-03', '1997-10-09', 3, 398.359985, 'quick-stop', 'taucherstraße 10', 'cunewalde', null, '01307', 'germany');
insert into orders values (10695, 'wilmk', 7, '1997-10-07', '1997-11-18', '1997-10-14', 1, 16.7199993, 'wilman kala', 'keskuskatu 45', 'helsinki', null, '21240', 'finland');
insert into orders values (10696, 'whitc', 8, '1997-10-08', '1997-11-19', '1997-10-14', 3, 102.550003, 'white clover markets', '1029 - 12th ave. s.', 'seattle', 'wa', '98124', 'usa');
insert into orders values (10697, 'linod', 3, '1997-10-08', '1997-11-05', '1997-10-14', 1, 45.5200005, 'lino-delicateses', 'ave. 5 de mayo porlamar', 'i. de margarita', 'nueva esparta', '4980', 'venezuela');
insert into orders values (10698, 'ernsh', 4, '1997-10-09', '1997-11-06', '1997-10-17', 1, 272.470001, 'ernst handel', 'kirchgasse 6', 'graz', null, '8010', 'austria');
insert into orders values (10699, 'morgk', 3, '1997-10-09', '1997-11-06', '1997-10-13', 3, 0.579999983, 'morgenstern gesundkost', 'heerstr. 22', 'leipzig', null, '04179', 'germany');
insert into orders values (10700, 'savea', 3, '1997-10-10', '1997-11-07', '1997-10-16', 1, 65.0999985, 'save-a-lot markets', '187 suffolk ln.', 'boise', 'id', '83720', 'usa');
insert into orders values (10701, 'hungo', 6, '1997-10-13', '1997-10-27', '1997-10-15', 3, 220.309998, 'hungry owl all-night grocers', '8 johnstown road', 'cork', 'co. cork', null, 'ireland');
insert into orders values (10702, 'alfki', 4, '1997-10-13', '1997-11-24', '1997-10-21', 1, 23.9400005, 'alfred''s futterkiste', 'obere str. 57', 'berlin', null, '12209', 'germany');
insert into orders values (10703, 'folko', 6, '1997-10-14', '1997-11-11', '1997-10-20', 2, 152.300003, 'folk och fä hb', 'åkergatan 24', 'bräcke', null, 's-844 67', 'sweden');
insert into orders values (10704, 'queen', 6, '1997-10-14', '1997-11-11', '1997-11-07', 1, 4.78000021, 'queen cozinha', 'alameda dos canàrios, 891', 'sao paulo', 'sp', '05487-020', 'brazil');
insert into orders values (10705, 'hilaa', 9, '1997-10-15', '1997-11-12', '1997-11-18', 2, 3.51999998, 'hilarion-abastos', 'carrera 22 con ave. carlos soublette #8-35', 'san cristóbal', 'táchira', '5022', 'venezuela');
insert into orders values (10706, 'oldwo', 8, '1997-10-16', '1997-11-13', '1997-10-21', 3, 135.630005, 'old world delicatessen', '2743 bering st.', 'anchorage', 'ak', '99508', 'usa');
insert into orders values (10707, 'arout', 4, '1997-10-16', '1997-10-30', '1997-10-23', 3, 21.7399998, 'around the horn', 'brook farm stratford st. mary', 'colchester', 'essex', 'co7 6jx', 'uk');
insert into orders values (10708, 'thebi', 6, '1997-10-17', '1997-11-28', '1997-11-05', 2, 2.96000004, 'the big cheese', '89 jefferson way suite 2', 'portland', 'or', '97201', 'usa');
insert into orders values (10709, 'gourl', 1, '1997-10-17', '1997-11-14', '1997-11-20', 3, 210.800003, 'gourmet lanchonetes', 'av. brasil, 442', 'campinas', 'sp', '04876-786', 'brazil');
insert into orders values (10710, 'frans', 1, '1997-10-20', '1997-11-17', '1997-10-23', 1, 4.98000002, 'franchi s.p.a.', 'via monte bianco 34', 'torino', null, '10100', 'italy');
insert into orders values (10711, 'savea', 5, '1997-10-21', '1997-12-02', '1997-10-29', 2, 52.4099998, 'save-a-lot markets', '187 suffolk ln.', 'boise', 'id', '83720', 'usa');
insert into orders values (10712, 'hungo', 3, '1997-10-21', '1997-11-18', '1997-10-31', 1, 89.9300003, 'hungry owl all-night grocers', '8 johnstown road', 'cork', 'co. cork', null, 'ireland');
insert into orders values (10713, 'savea', 1, '1997-10-22', '1997-11-19', '1997-10-24', 1, 167.050003, 'save-a-lot markets', '187 suffolk ln.', 'boise', 'id', '83720', 'usa');
insert into orders values (10714, 'savea', 5, '1997-10-22', '1997-11-19', '1997-10-27', 3, 24.4899998, 'save-a-lot markets', '187 suffolk ln.', 'boise', 'id', '83720', 'usa');
insert into orders values (10715, 'bonap', 3, '1997-10-23', '1997-11-06', '1997-10-29', 1, 63.2000008, 'bon app''', '12, rue des bouchers', 'marseille', null, '13008', 'france');
insert into orders values (10716, 'ranch', 4, '1997-10-24', '1997-11-21', '1997-10-27', 2, 22.5699997, 'rancho grande', 'av. del libertador 900', 'buenos aires', null, '1010', 'argentina');
insert into orders values (10717, 'frank', 1, '1997-10-24', '1997-11-21', '1997-10-29', 2, 59.25, 'frankenversand', 'berliner platz 43', 'münchen', null, '80805', 'germany');
insert into orders values (10718, 'koene', 1, '1997-10-27', '1997-11-24', '1997-10-29', 3, 170.880005, 'königlich essen', 'maubelstr. 90', 'brandenburg', null, '14776', 'germany');
insert into orders values (10719, 'letss', 8, '1997-10-27', '1997-11-24', '1997-11-05', 2, 51.4399986, 'let''s stop n shop', '87 polk st. suite 5', 'san francisco', 'ca', '94117', 'usa');
insert into orders values (10720, 'quede', 8, '1997-10-28', '1997-11-11', '1997-11-05', 2, 9.52999973, 'que delícia', 'rua da panificadora, 12', 'rio de janeiro', 'rj', '02389-673', 'brazil');
insert into orders values (10721, 'quick', 5, '1997-10-29', '1997-11-26', '1997-10-31', 3, 48.9199982, 'quick-stop', 'taucherstraße 10', 'cunewalde', null, '01307', 'germany');
insert into orders values (10722, 'savea', 8, '1997-10-29', '1997-12-10', '1997-11-04', 1, 74.5800018, 'save-a-lot markets', '187 suffolk ln.', 'boise', 'id', '83720', 'usa');
insert into orders values (10723, 'whitc', 3, '1997-10-30', '1997-11-27', '1997-11-25', 1, 21.7199993, 'white clover markets', '1029 - 12th ave. s.', 'seattle', 'wa', '98124', 'usa');
insert into orders values (10724, 'merep', 8, '1997-10-30', '1997-12-11', '1997-11-05', 2, 57.75, 'mère paillarde', '43 rue st. laurent', 'montréal', 'québec', 'h1j 1c3', 'canada');
insert into orders values (10725, 'famia', 4, '1997-10-31', '1997-11-28', '1997-11-05', 3, 10.8299999, 'familia arquibaldo', 'rua orós, 92', 'sao paulo', 'sp', '05442-030', 'brazil');
insert into orders values (10726, 'eastc', 4, '1997-11-03', '1997-11-17', '1997-12-05', 1, 16.5599995, 'eastern connection', '35 king george', 'london', null, 'wx3 6fw', 'uk');
insert into orders values (10727, 'reggc', 2, '1997-11-03', '1997-12-01', '1997-12-05', 1, 89.9000015, 'reggiani caseifici', 'strada provinciale 124', 'reggio emilia', null, '42100', 'italy');
insert into orders values (10728, 'queen', 4, '1997-11-04', '1997-12-02', '1997-11-11', 2, 58.3300018, 'queen cozinha', 'alameda dos canàrios, 891', 'sao paulo', 'sp', '05487-020', 'brazil');
insert into orders values (10729, 'linod', 8, '1997-11-04', '1997-12-16', '1997-11-14', 3, 141.059998, 'lino-delicateses', 'ave. 5 de mayo porlamar', 'i. de margarita', 'nueva esparta', '4980', 'venezuela');
insert into orders values (10730, 'bonap', 5, '1997-11-05', '1997-12-03', '1997-11-14', 1, 20.1200008, 'bon app''', '12, rue des bouchers', 'marseille', null, '13008', 'france');
insert into orders values (10731, 'chops', 7, '1997-11-06', '1997-12-04', '1997-11-14', 1, 96.6500015, 'chop-suey chinese', 'hauptstr. 31', 'bern', null, '3012', 'switzerland');
insert into orders values (10732, 'bonap', 3, '1997-11-06', '1997-12-04', '1997-11-07', 1, 16.9699993, 'bon app''', '12, rue des bouchers', 'marseille', null, '13008', 'france');
insert into orders values (10733, 'bergs', 1, '1997-11-07', '1997-12-05', '1997-11-10', 3, 110.110001, 'berglunds snabbköp', 'berguvsvägen  8', 'luleå', null, 's-958 22', 'sweden');
insert into orders values (10734, 'gourl', 2, '1997-11-07', '1997-12-05', '1997-11-12', 3, 1.63, 'gourmet lanchonetes', 'av. brasil, 442', 'campinas', 'sp', '04876-786', 'brazil');
insert into orders values (10735, 'letss', 6, '1997-11-10', '1997-12-08', '1997-11-21', 2, 45.9700012, 'let''s stop n shop', '87 polk st. suite 5', 'san francisco', 'ca', '94117', 'usa');
insert into orders values (10736, 'hungo', 9, '1997-11-11', '1997-12-09', '1997-11-21', 2, 44.0999985, 'hungry owl all-night grocers', '8 johnstown road', 'cork', 'co. cork', null, 'ireland');
insert into orders values (10737, 'vinet', 2, '1997-11-11', '1997-12-09', '1997-11-18', 2, 7.78999996, 'vins et alcools chevalier', '59 rue de l''abbaye', 'reims', null, '51100', 'france');
insert into orders values (10738, 'specd', 2, '1997-11-12', '1997-12-10', '1997-11-18', 1, 2.91000009, 'spécialités du monde', '25, rue lauriston', 'paris', null, '75016', 'france');
insert into orders values (10739, 'vinet', 3, '1997-11-12', '1997-12-10', '1997-11-17', 3, 11.0799999, 'vins et alcools chevalier', '59 rue de l''abbaye', 'reims', null, '51100', 'france');
insert into orders values (10740, 'whitc', 4, '1997-11-13', '1997-12-11', '1997-11-25', 2, 81.8799973, 'white clover markets', '1029 - 12th ave. s.', 'seattle', 'wa', '98124', 'usa');
insert into orders values (10741, 'arout', 4, '1997-11-14', '1997-11-28', '1997-11-18', 3, 10.96, 'around the horn', 'brook farm stratford st. mary', 'colchester', 'essex', 'co7 6jx', 'uk');
insert into orders values (10742, 'bottm', 3, '1997-11-14', '1997-12-12', '1997-11-18', 3, 243.729996, 'bottom-dollar markets', '23 tsawassen blvd.', 'tsawassen', 'bc', 't2f 8m4', 'canada');
insert into orders values (10743, 'arout', 1, '1997-11-17', '1997-12-15', '1997-11-21', 2, 23.7199993, 'around the horn', 'brook farm stratford st. mary', 'colchester', 'essex', 'co7 6jx', 'uk');
insert into orders values (10744, 'vaffe', 6, '1997-11-17', '1997-12-15', '1997-11-24', 1, 69.1900024, 'vaffeljernet', 'smagsloget 45', 'århus', null, '8200', 'denmark');
insert into orders values (10745, 'quick', 9, '1997-11-18', '1997-12-16', '1997-11-27', 1, 3.51999998, 'quick-stop', 'taucherstraße 10', 'cunewalde', null, '01307', 'germany');
insert into orders values (10746, 'chops', 1, '1997-11-19', '1997-12-17', '1997-11-21', 3, 31.4300003, 'chop-suey chinese', 'hauptstr. 31', 'bern', null, '3012', 'switzerland');
insert into orders values (10747, 'picco', 6, '1997-11-19', '1997-12-17', '1997-11-26', 1, 117.330002, 'piccolo und mehr', 'geislweg 14', 'salzburg', null, '5020', 'austria');
insert into orders values (10748, 'savea', 3, '1997-11-20', '1997-12-18', '1997-11-28', 1, 232.550003, 'save-a-lot markets', '187 suffolk ln.', 'boise', 'id', '83720', 'usa');
insert into orders values (10749, 'islat', 4, '1997-11-20', '1997-12-18', '1997-12-19', 2, 61.5299988, 'island trading', 'garden house crowther way', 'cowes', 'isle of wight', 'po31 7pj', 'uk');
insert into orders values (10750, 'warth', 9, '1997-11-21', '1997-12-19', '1997-11-24', 1, 79.3000031, 'wartian herkku', 'torikatu 38', 'oulu', null, '90110', 'finland');
insert into orders values (10751, 'ricsu', 3, '1997-11-24', '1997-12-22', '1997-12-03', 3, 130.789993, 'richter supermarkt', 'starenweg 5', 'genève', null, '1204', 'switzerland');
insert into orders values (10752, 'norts', 2, '1997-11-24', '1997-12-22', '1997-11-28', 3, 1.38999999, 'north/south', 'south house 300 queensbridge', 'london', null, 'sw7 1rz', 'uk');
insert into orders values (10753, 'frans', 3, '1997-11-25', '1997-12-23', '1997-11-27', 1, 7.69999981, 'franchi s.p.a.', 'via monte bianco 34', 'torino', null, '10100', 'italy');
insert into orders values (10754, 'magaa', 6, '1997-11-25', '1997-12-23', '1997-11-27', 3, 2.38000011, 'magazzini alimentari riuniti', 'via ludovico il moro 22', 'bergamo', null, '24100', 'italy');
insert into orders values (10755, 'bonap', 4, '1997-11-26', '1997-12-24', '1997-11-28', 2, 16.7099991, 'bon app''', '12, rue des bouchers', 'marseille', null, '13008', 'france');
insert into orders values (10756, 'splir', 8, '1997-11-27', '1997-12-25', '1997-12-02', 2, 73.2099991, 'split rail beer & ale', 'p.o. box 555', 'lander', 'wy', '82520', 'usa');
insert into orders values (10757, 'savea', 6, '1997-11-27', '1997-12-25', '1997-12-15', 1, 8.18999958, 'save-a-lot markets', '187 suffolk ln.', 'boise', 'id', '83720', 'usa');
insert into orders values (10758, 'ricsu', 3, '1997-11-28', '1997-12-26', '1997-12-04', 3, 138.169998, 'richter supermarkt', 'starenweg 5', 'genève', null, '1204', 'switzerland');
insert into orders values (10759, 'anatr', 3, '1997-11-28', '1997-12-26', '1997-12-12', 3, 11.9899998, 'ana trujillo emparedados y helados', 'avda. de la constitución 2222', 'méxico d.f.', null, '05021', 'mexico');
insert into orders values (10760, 'maisd', 4, '1997-12-01', '1997-12-29', '1997-12-10', 1, 155.639999, 'maison dewey', 'rue joseph-bens 532', 'bruxelles', null, 'b-1180', 'belgium');
insert into orders values (10761, 'rattc', 5, '1997-12-02', '1997-12-30', '1997-12-08', 2, 18.6599998, 'rattlesnake canyon grocery', '2817 milton dr.', 'albuquerque', 'nm', '87110', 'usa');
insert into orders values (10762, 'folko', 3, '1997-12-02', '1997-12-30', '1997-12-09', 1, 328.73999, 'folk och fä hb', 'åkergatan 24', 'bräcke', null, 's-844 67', 'sweden');
insert into orders values (10763, 'folig', 3, '1997-12-03', '1997-12-31', '1997-12-08', 3, 37.3499985, 'folies gourmandes', '184, chaussée de tournai', 'lille', null, '59000', 'france');
insert into orders values (10764, 'ernsh', 6, '1997-12-03', '1997-12-31', '1997-12-08', 3, 145.449997, 'ernst handel', 'kirchgasse 6', 'graz', null, '8010', 'austria');
insert into orders values (10765, 'quick', 3, '1997-12-04', '1998-01-01', '1997-12-09', 3, 42.7400017, 'quick-stop', 'taucherstraße 10', 'cunewalde', null, '01307', 'germany');
insert into orders values (10766, 'ottik', 4, '1997-12-05', '1998-01-02', '1997-12-09', 1, 157.550003, 'ottilies käseladen', 'mehrheimerstr. 369', 'köln', null, '50739', 'germany');
insert into orders values (10767, 'suprd', 4, '1997-12-05', '1998-01-02', '1997-12-15', 3, 1.59000003, 'suprêmes délices', 'boulevard tirou, 255', 'charleroi', null, 'b-6000', 'belgium');
insert into orders values (10768, 'arout', 3, '1997-12-08', '1998-01-05', '1997-12-15', 2, 146.320007, 'around the horn', 'brook farm stratford st. mary', 'colchester', 'essex', 'co7 6jx', 'uk');
insert into orders values (10769, 'vaffe', 3, '1997-12-08', '1998-01-05', '1997-12-12', 1, 65.0599976, 'vaffeljernet', 'smagsloget 45', 'århus', null, '8200', 'denmark');
insert into orders values (10770, 'hanar', 8, '1997-12-09', '1998-01-06', '1997-12-17', 3, 5.32000017, 'hanari carnes', 'rua do paço, 67', 'rio de janeiro', 'rj', '05454-876', 'brazil');
insert into orders values (10771, 'ernsh', 9, '1997-12-10', '1998-01-07', '1998-01-02', 2, 11.1899996, 'ernst handel', 'kirchgasse 6', 'graz', null, '8010', 'austria');
insert into orders values (10772, 'lehms', 3, '1997-12-10', '1998-01-07', '1997-12-19', 2, 91.2799988, 'lehmanns marktstand', 'magazinweg 7', 'frankfurt a.m.', null, '60528', 'germany');
insert into orders values (10773, 'ernsh', 1, '1997-12-11', '1998-01-08', '1997-12-16', 3, 96.4300003, 'ernst handel', 'kirchgasse 6', 'graz', null, '8010', 'austria');
insert into orders values (10774, 'folko', 4, '1997-12-11', '1997-12-25', '1997-12-12', 1, 48.2000008, 'folk och fä hb', 'åkergatan 24', 'bräcke', null, 's-844 67', 'sweden');
insert into orders values (10775, 'thecr', 7, '1997-12-12', '1998-01-09', '1997-12-26', 1, 20.25, 'the cracker box', '55 grizzly peak rd.', 'butte', 'mt', '59801', 'usa');
insert into orders values (10776, 'ernsh', 1, '1997-12-15', '1998-01-12', '1997-12-18', 3, 351.529999, 'ernst handel', 'kirchgasse 6', 'graz', null, '8010', 'austria');
insert into orders values (10777, 'gourl', 7, '1997-12-15', '1997-12-29', '1998-01-21', 2, 3.00999999, 'gourmet lanchonetes', 'av. brasil, 442', 'campinas', 'sp', '04876-786', 'brazil');
insert into orders values (10778, 'bergs', 3, '1997-12-16', '1998-01-13', '1997-12-24', 1, 6.78999996, 'berglunds snabbköp', 'berguvsvägen  8', 'luleå', null, 's-958 22', 'sweden');
insert into orders values (10779, 'morgk', 3, '1997-12-16', '1998-01-13', '1998-01-14', 2, 58.1300011, 'morgenstern gesundkost', 'heerstr. 22', 'leipzig', null, '04179', 'germany');
insert into orders values (10780, 'lilas', 2, '1997-12-16', '1997-12-30', '1997-12-25', 1, 42.1300011, 'lila-supermercado', 'carrera 52 con ave. bolívar #65-98 llano largo', 'barquisimeto', 'lara', '3508', 'venezuela');
insert into orders values (10781, 'warth', 2, '1997-12-17', '1998-01-14', '1997-12-19', 3, 73.1600037, 'wartian herkku', 'torikatu 38', 'oulu', null, '90110', 'finland');
insert into orders values (10782, 'cactu', 9, '1997-12-17', '1998-01-14', '1997-12-22', 3, 1.10000002, 'cactus comidas para llevar', 'cerrito 333', 'buenos aires', null, '1010', 'argentina');
insert into orders values (10783, 'hanar', 4, '1997-12-18', '1998-01-15', '1997-12-19', 2, 124.980003, 'hanari carnes', 'rua do paço, 67', 'rio de janeiro', 'rj', '05454-876', 'brazil');
insert into orders values (10784, 'magaa', 4, '1997-12-18', '1998-01-15', '1997-12-22', 3, 70.0899963, 'magazzini alimentari riuniti', 'via ludovico il moro 22', 'bergamo', null, '24100', 'italy');
insert into orders values (10785, 'grosr', 1, '1997-12-18', '1998-01-15', '1997-12-24', 3, 1.50999999, 'grosella-restaurante', '5ª ave. los palos grandes', 'caracas', 'df', '1081', 'venezuela');
insert into orders values (10786, 'queen', 8, '1997-12-19', '1998-01-16', '1997-12-23', 1, 110.870003, 'queen cozinha', 'alameda dos canàrios, 891', 'sao paulo', 'sp', '05487-020', 'brazil');
insert into orders values (10787, 'lamai', 2, '1997-12-19', '1998-01-02', '1997-12-26', 1, 249.929993, 'la maison d''asie', '1 rue alsace-lorraine', 'toulouse', null, '31000', 'france');
insert into orders values (10788, 'quick', 1, '1997-12-22', '1998-01-19', '1998-01-19', 2, 42.7000008, 'quick-stop', 'taucherstraße 10', 'cunewalde', null, '01307', 'germany');
insert into orders values (10789, 'folig', 1, '1997-12-22', '1998-01-19', '1997-12-31', 2, 100.599998, 'folies gourmandes', '184, chaussée de tournai', 'lille', null, '59000', 'france');
insert into orders values (10790, 'gourl', 6, '1997-12-22', '1998-01-19', '1997-12-26', 1, 28.2299995, 'gourmet lanchonetes', 'av. brasil, 442', 'campinas', 'sp', '04876-786', 'brazil');
insert into orders values (10791, 'frank', 6, '1997-12-23', '1998-01-20', '1998-01-01', 2, 16.8500004, 'frankenversand', 'berliner platz 43', 'münchen', null, '80805', 'germany');
insert into orders values (10792, 'wolza', 1, '1997-12-23', '1998-01-20', '1997-12-31', 3, 23.7900009, 'wolski zajazd', 'ul. filtrowa 68', 'warszawa', null, '01-012', 'poland');
insert into orders values (10793, 'arout', 3, '1997-12-24', '1998-01-21', '1998-01-08', 3, 4.51999998, 'around the horn', 'brook farm stratford st. mary', 'colchester', 'essex', 'co7 6jx', 'uk');
insert into orders values (10794, 'quede', 6, '1997-12-24', '1998-01-21', '1998-01-02', 1, 21.4899998, 'que delícia', 'rua da panificadora, 12', 'rio de janeiro', 'rj', '02389-673', 'brazil');
insert into orders values (10795, 'ernsh', 8, '1997-12-24', '1998-01-21', '1998-01-20', 2, 126.660004, 'ernst handel', 'kirchgasse 6', 'graz', null, '8010', 'austria');
insert into orders values (10796, 'hilaa', 3, '1997-12-25', '1998-01-22', '1998-01-14', 1, 26.5200005, 'hilarion-abastos', 'carrera 22 con ave. carlos soublette #8-35', 'san cristóbal', 'táchira', '5022', 'venezuela');
insert into orders values (10797, 'dracd', 7, '1997-12-25', '1998-01-22', '1998-01-05', 2, 33.3499985, 'drachenblut delikatessen', 'walserweg 21', 'aachen', null, '52066', 'germany');
insert into orders values (10798, 'islat', 2, '1997-12-26', '1998-01-23', '1998-01-05', 1, 2.32999992, 'island trading', 'garden house crowther way', 'cowes', 'isle of wight', 'po31 7pj', 'uk');
insert into orders values (10799, 'koene', 9, '1997-12-26', '1998-02-06', '1998-01-05', 3, 30.7600002, 'königlich essen', 'maubelstr. 90', 'brandenburg', null, '14776', 'germany');
insert into orders values (10800, 'seves', 1, '1997-12-26', '1998-01-23', '1998-01-05', 3, 137.440002, 'seven seas imports', '90 wadhurst rd.', 'london', null, 'ox15 4nb', 'uk');
insert into orders values (10801, 'bolid', 4, '1997-12-29', '1998-01-26', '1997-12-31', 2, 97.0899963, 'bólido comidas preparadas', 'c/ araquil, 67', 'madrid', null, '28023', 'spain');
insert into orders values (10802, 'simob', 4, '1997-12-29', '1998-01-26', '1998-01-02', 2, 257.26001, 'simons bistro', 'vinbæltet 34', 'kobenhavn', null, '1734', 'denmark');
insert into orders values (10803, 'welli', 4, '1997-12-30', '1998-01-27', '1998-01-06', 1, 55.2299995, 'wellington importadora', 'rua do mercado, 12', 'resende', 'sp', '08737-363', 'brazil');
insert into orders values (10804, 'seves', 6, '1997-12-30', '1998-01-27', '1998-01-07', 2, 27.3299999, 'seven seas imports', '90 wadhurst rd.', 'london', null, 'ox15 4nb', 'uk');
insert into orders values (10805, 'thebi', 2, '1997-12-30', '1998-01-27', '1998-01-09', 3, 237.339996, 'the big cheese', '89 jefferson way suite 2', 'portland', 'or', '97201', 'usa');
insert into orders values (10806, 'victe', 3, '1997-12-31', '1998-01-28', '1998-01-05', 2, 22.1100006, 'victuailles en stock', '2, rue du commerce', 'lyon', null, '69004', 'france');
insert into orders values (10807, 'frans', 4, '1997-12-31', '1998-01-28', '1998-01-30', 1, 1.36000001, 'franchi s.p.a.', 'via monte bianco 34', 'torino', null, '10100', 'italy');
insert into orders values (10808, 'oldwo', 2, '1998-01-01', '1998-01-29', '1998-01-09', 3, 45.5299988, 'old world delicatessen', '2743 bering st.', 'anchorage', 'ak', '99508', 'usa');
insert into orders values (10809, 'welli', 7, '1998-01-01', '1998-01-29', '1998-01-07', 1, 4.86999989, 'wellington importadora', 'rua do mercado, 12', 'resende', 'sp', '08737-363', 'brazil');
insert into orders values (10810, 'laugb', 2, '1998-01-01', '1998-01-29', '1998-01-07', 3, 4.32999992, 'laughing bacchus wine cellars', '2319 elm st.', 'vancouver', 'bc', 'v3f 2k1', 'canada');
insert into orders values (10811, 'linod', 8, '1998-01-02', '1998-01-30', '1998-01-08', 1, 31.2199993, 'lino-delicateses', 'ave. 5 de mayo porlamar', 'i. de margarita', 'nueva esparta', '4980', 'venezuela');
insert into orders values (10812, 'reggc', 5, '1998-01-02', '1998-01-30', '1998-01-12', 1, 59.7799988, 'reggiani caseifici', 'strada provinciale 124', 'reggio emilia', null, '42100', 'italy');
insert into orders values (10813, 'ricar', 1, '1998-01-05', '1998-02-02', '1998-01-09', 1, 47.3800011, 'ricardo adocicados', 'av. copacabana, 267', 'rio de janeiro', 'rj', '02389-890', 'brazil');
insert into orders values (10814, 'victe', 3, '1998-01-05', '1998-02-02', '1998-01-14', 3, 130.940002, 'victuailles en stock', '2, rue du commerce', 'lyon', null, '69004', 'france');
insert into orders values (10815, 'savea', 2, '1998-01-05', '1998-02-02', '1998-01-14', 3, 14.6199999, 'save-a-lot markets', '187 suffolk ln.', 'boise', 'id', '83720', 'usa');
insert into orders values (10816, 'greal', 4, '1998-01-06', '1998-02-03', '1998-02-04', 2, 719.780029, 'great lakes food market', '2732 baker blvd.', 'eugene', 'or', '97403', 'usa');
insert into orders values (10817, 'koene', 3, '1998-01-06', '1998-01-20', '1998-01-13', 2, 306.070007, 'königlich essen', 'maubelstr. 90', 'brandenburg', null, '14776', 'germany');
insert into orders values (10818, 'magaa', 7, '1998-01-07', '1998-02-04', '1998-01-12', 3, 65.4800034, 'magazzini alimentari riuniti', 'via ludovico il moro 22', 'bergamo', null, '24100', 'italy');
insert into orders values (10819, 'cactu', 2, '1998-01-07', '1998-02-04', '1998-01-16', 3, 19.7600002, 'cactus comidas para llevar', 'cerrito 333', 'buenos aires', null, '1010', 'argentina');
insert into orders values (10820, 'rattc', 3, '1998-01-07', '1998-02-04', '1998-01-13', 2, 37.5200005, 'rattlesnake canyon grocery', '2817 milton dr.', 'albuquerque', 'nm', '87110', 'usa');
insert into orders values (10821, 'splir', 1, '1998-01-08', '1998-02-05', '1998-01-15', 1, 36.6800003, 'split rail beer & ale', 'p.o. box 555', 'lander', 'wy', '82520', 'usa');
insert into orders values (10822, 'traih', 6, '1998-01-08', '1998-02-05', '1998-01-16', 3, 7, 'trail''s head gourmet provisioners', '722 davinci blvd.', 'kirkland', 'wa', '98034', 'usa');
insert into orders values (10823, 'lilas', 5, '1998-01-09', '1998-02-06', '1998-01-13', 2, 163.970001, 'lila-supermercado', 'carrera 52 con ave. bolívar #65-98 llano largo', 'barquisimeto', 'lara', '3508', 'venezuela');
insert into orders values (10824, 'folko', 8, '1998-01-09', '1998-02-06', '1998-01-30', 1, 1.23000002, 'folk och fä hb', 'åkergatan 24', 'bräcke', null, 's-844 67', 'sweden');
insert into orders values (10825, 'dracd', 1, '1998-01-09', '1998-02-06', '1998-01-14', 1, 79.25, 'drachenblut delikatessen', 'walserweg 21', 'aachen', null, '52066', 'germany');
insert into orders values (10826, 'blonp', 6, '1998-01-12', '1998-02-09', '1998-02-06', 1, 7.09000015, 'blondel père et fils', '24, place kléber', 'strasbourg', null, '67000', 'france');
insert into orders values (10827, 'bonap', 1, '1998-01-12', '1998-01-26', '1998-02-06', 2, 63.5400009, 'bon app''', '12, rue des bouchers', 'marseille', null, '13008', 'france');
insert into orders values (10828, 'ranch', 9, '1998-01-13', '1998-01-27', '1998-02-04', 1, 90.8499985, 'rancho grande', 'av. del libertador 900', 'buenos aires', null, '1010', 'argentina');
insert into orders values (10829, 'islat', 9, '1998-01-13', '1998-02-10', '1998-01-23', 1, 154.720001, 'island trading', 'garden house crowther way', 'cowes', 'isle of wight', 'po31 7pj', 'uk');
insert into orders values (10830, 'tradh', 4, '1998-01-13', '1998-02-24', '1998-01-21', 2, 81.8300018, 'tradiçao hipermercados', 'av. inês de castro, 414', 'sao paulo', 'sp', '05634-030', 'brazil');
insert into orders values (10831, 'santg', 3, '1998-01-14', '1998-02-11', '1998-01-23', 2, 72.1900024, 'santé gourmet', 'erling skakkes gate 78', 'stavern', null, '4110', 'norway');
insert into orders values (10832, 'lamai', 2, '1998-01-14', '1998-02-11', '1998-01-19', 2, 43.2599983, 'la maison d''asie', '1 rue alsace-lorraine', 'toulouse', null, '31000', 'france');
insert into orders values (10833, 'ottik', 6, '1998-01-15', '1998-02-12', '1998-01-23', 2, 71.4899979, 'ottilies käseladen', 'mehrheimerstr. 369', 'köln', null, '50739', 'germany');
insert into orders values (10834, 'tradh', 1, '1998-01-15', '1998-02-12', '1998-01-19', 3, 29.7800007, 'tradiçao hipermercados', 'av. inês de castro, 414', 'sao paulo', 'sp', '05634-030', 'brazil');
insert into orders values (10835, 'alfki', 1, '1998-01-15', '1998-02-12', '1998-01-21', 3, 69.5299988, 'alfred''s futterkiste', 'obere str. 57', 'berlin', null, '12209', 'germany');
insert into orders values (10836, 'ernsh', 7, '1998-01-16', '1998-02-13', '1998-01-21', 1, 411.880005, 'ernst handel', 'kirchgasse 6', 'graz', null, '8010', 'austria');
insert into orders values (10837, 'bergs', 9, '1998-01-16', '1998-02-13', '1998-01-23', 3, 13.3199997, 'berglunds snabbköp', 'berguvsvägen  8', 'luleå', null, 's-958 22', 'sweden');
insert into orders values (10838, 'linod', 3, '1998-01-19', '1998-02-16', '1998-01-23', 3, 59.2799988, 'lino-delicateses', 'ave. 5 de mayo porlamar', 'i. de margarita', 'nueva esparta', '4980', 'venezuela');
insert into orders values (10839, 'tradh', 3, '1998-01-19', '1998-02-16', '1998-01-22', 3, 35.4300003, 'tradiçao hipermercados', 'av. inês de castro, 414', 'sao paulo', 'sp', '05634-030', 'brazil');
insert into orders values (10840, 'linod', 4, '1998-01-19', '1998-03-02', '1998-02-16', 2, 2.71000004, 'lino-delicateses', 'ave. 5 de mayo porlamar', 'i. de margarita', 'nueva esparta', '4980', 'venezuela');
insert into orders values (10841, 'suprd', 5, '1998-01-20', '1998-02-17', '1998-01-29', 2, 424.299988, 'suprêmes délices', 'boulevard tirou, 255', 'charleroi', null, 'b-6000', 'belgium');
insert into orders values (10842, 'tortu', 1, '1998-01-20', '1998-02-17', '1998-01-29', 3, 54.4199982, 'tortuga restaurante', 'avda. azteca 123', 'méxico d.f.', null, '05033', 'mexico');
insert into orders values (10843, 'victe', 4, '1998-01-21', '1998-02-18', '1998-01-26', 2, 9.26000023, 'victuailles en stock', '2, rue du commerce', 'lyon', null, '69004', 'france');
insert into orders values (10844, 'picco', 8, '1998-01-21', '1998-02-18', '1998-01-26', 2, 25.2199993, 'piccolo und mehr', 'geislweg 14', 'salzburg', null, '5020', 'austria');
insert into orders values (10845, 'quick', 8, '1998-01-21', '1998-02-04', '1998-01-30', 1, 212.979996, 'quick-stop', 'taucherstraße 10', 'cunewalde', null, '01307', 'germany');
insert into orders values (10846, 'suprd', 2, '1998-01-22', '1998-03-05', '1998-01-23', 3, 56.4599991, 'suprêmes délices', 'boulevard tirou, 255', 'charleroi', null, 'b-6000', 'belgium');
insert into orders values (10847, 'savea', 4, '1998-01-22', '1998-02-05', '1998-02-10', 3, 487.570007, 'save-a-lot markets', '187 suffolk ln.', 'boise', 'id', '83720', 'usa');
insert into orders values (10848, 'consh', 7, '1998-01-23', '1998-02-20', '1998-01-29', 2, 38.2400017, 'consolidated holdings', 'berkeley gardens 12  brewery', 'london', null, 'wx1 6lt', 'uk');
insert into orders values (10849, 'koene', 9, '1998-01-23', '1998-02-20', '1998-01-30', 2, 0.560000002, 'königlich essen', 'maubelstr. 90', 'brandenburg', null, '14776', 'germany');
insert into orders values (10850, 'victe', 1, '1998-01-23', '1998-03-06', '1998-01-30', 1, 49.1899986, 'victuailles en stock', '2, rue du commerce', 'lyon', null, '69004', 'france');
insert into orders values (10851, 'ricar', 5, '1998-01-26', '1998-02-23', '1998-02-02', 1, 160.550003, 'ricardo adocicados', 'av. copacabana, 267', 'rio de janeiro', 'rj', '02389-890', 'brazil');
insert into orders values (10852, 'rattc', 8, '1998-01-26', '1998-02-09', '1998-01-30', 1, 174.050003, 'rattlesnake canyon grocery', '2817 milton dr.', 'albuquerque', 'nm', '87110', 'usa');
insert into orders values (10853, 'blaus', 9, '1998-01-27', '1998-02-24', '1998-02-03', 2, 53.8300018, 'blauer see delikatessen', 'forsterstr. 57', 'mannheim', null, '68306', 'germany');
insert into orders values (10854, 'ernsh', 3, '1998-01-27', '1998-02-24', '1998-02-05', 2, 100.220001, 'ernst handel', 'kirchgasse 6', 'graz', null, '8010', 'austria');
insert into orders values (10855, 'oldwo', 3, '1998-01-27', '1998-02-24', '1998-02-04', 1, 170.970001, 'old world delicatessen', '2743 bering st.', 'anchorage', 'ak', '99508', 'usa');
insert into orders values (10856, 'anton', 3, '1998-01-28', '1998-02-25', '1998-02-10', 2, 58.4300003, 'antonio moreno taquería', 'mataderos  2312', 'méxico d.f.', null, '05023', 'mexico');
insert into orders values (10857, 'bergs', 8, '1998-01-28', '1998-02-25', '1998-02-06', 2, 188.850006, 'berglunds snabbköp', 'berguvsvägen  8', 'luleå', null, 's-958 22', 'sweden');
insert into orders values (10858, 'lacor', 2, '1998-01-29', '1998-02-26', '1998-02-03', 1, 52.5099983, 'la corne d''abondance', '67, avenue de l''europe', 'versailles', null, '78000', 'france');
insert into orders values (10859, 'frank', 1, '1998-01-29', '1998-02-26', '1998-02-02', 2, 76.0999985, 'frankenversand', 'berliner platz 43', 'münchen', null, '80805', 'germany');
insert into orders values (10860, 'franr', 3, '1998-01-29', '1998-02-26', '1998-02-04', 3, 19.2600002, 'france restauration', '54, rue royale', 'nantes', null, '44000', 'france');
insert into orders values (10861, 'whitc', 4, '1998-01-30', '1998-02-27', '1998-02-17', 2, 14.9300003, 'white clover markets', '1029 - 12th ave. s.', 'seattle', 'wa', '98124', 'usa');
insert into orders values (10862, 'lehms', 8, '1998-01-30', '1998-03-13', '1998-02-02', 2, 53.2299995, 'lehmanns marktstand', 'magazinweg 7', 'frankfurt a.m.', null, '60528', 'germany');
insert into orders values (10863, 'hilaa', 4, '1998-02-02', '1998-03-02', '1998-02-17', 2, 30.2600002, 'hilarion-abastos', 'carrera 22 con ave. carlos soublette #8-35', 'san cristóbal', 'táchira', '5022', 'venezuela');
insert into orders values (10864, 'arout', 4, '1998-02-02', '1998-03-02', '1998-02-09', 2, 3.03999996, 'around the horn', 'brook farm stratford st. mary', 'colchester', 'essex', 'co7 6jx', 'uk');
insert into orders values (10865, 'quick', 2, '1998-02-02', '1998-02-16', '1998-02-12', 1, 348.140015, 'quick-stop', 'taucherstraße 10', 'cunewalde', null, '01307', 'germany');
insert into orders values (10866, 'bergs', 5, '1998-02-03', '1998-03-03', '1998-02-12', 1, 109.110001, 'berglunds snabbköp', 'berguvsvägen  8', 'luleå', null, 's-958 22', 'sweden');
insert into orders values (10867, 'lonep', 6, '1998-02-03', '1998-03-17', '1998-02-11', 1, 1.92999995, 'lonesome pine restaurant', '89 chiaroscuro rd.', 'portland', 'or', '97219', 'usa');
insert into orders values (10868, 'queen', 7, '1998-02-04', '1998-03-04', '1998-02-23', 2, 191.270004, 'queen cozinha', 'alameda dos canàrios, 891', 'sao paulo', 'sp', '05487-020', 'brazil');
insert into orders values (10869, 'seves', 5, '1998-02-04', '1998-03-04', '1998-02-09', 1, 143.279999, 'seven seas imports', '90 wadhurst rd.', 'london', null, 'ox15 4nb', 'uk');
insert into orders values (10870, 'wolza', 5, '1998-02-04', '1998-03-04', '1998-02-13', 3, 12.04, 'wolski zajazd', 'ul. filtrowa 68', 'warszawa', null, '01-012', 'poland');
insert into orders values (10871, 'bonap', 9, '1998-02-05', '1998-03-05', '1998-02-10', 2, 112.269997, 'bon app''', '12, rue des bouchers', 'marseille', null, '13008', 'france');
insert into orders values (10872, 'godos', 5, '1998-02-05', '1998-03-05', '1998-02-09', 2, 175.320007, 'godos cocina típica', 'c/ romero, 33', 'sevilla', null, '41101', 'spain');
insert into orders values (10873, 'wilmk', 4, '1998-02-06', '1998-03-06', '1998-02-09', 1, 0.819999993, 'wilman kala', 'keskuskatu 45', 'helsinki', null, '21240', 'finland');
insert into orders values (10874, 'godos', 5, '1998-02-06', '1998-03-06', '1998-02-11', 2, 19.5799999, 'godos cocina típica', 'c/ romero, 33', 'sevilla', null, '41101', 'spain');
insert into orders values (10875, 'bergs', 4, '1998-02-06', '1998-03-06', '1998-03-03', 2, 32.3699989, 'berglunds snabbköp', 'berguvsvägen  8', 'luleå', null, 's-958 22', 'sweden');
insert into orders values (10876, 'bonap', 7, '1998-02-09', '1998-03-09', '1998-02-12', 3, 60.4199982, 'bon app''', '12, rue des bouchers', 'marseille', null, '13008', 'france');
insert into orders values (10877, 'ricar', 1, '1998-02-09', '1998-03-09', '1998-02-19', 1, 38.0600014, 'ricardo adocicados', 'av. copacabana, 267', 'rio de janeiro', 'rj', '02389-890', 'brazil');
insert into orders values (10878, 'quick', 4, '1998-02-10', '1998-03-10', '1998-02-12', 1, 46.6899986, 'quick-stop', 'taucherstraße 10', 'cunewalde', null, '01307', 'germany');
insert into orders values (10879, 'wilmk', 3, '1998-02-10', '1998-03-10', '1998-02-12', 3, 8.5, 'wilman kala', 'keskuskatu 45', 'helsinki', null, '21240', 'finland');
insert into orders values (10880, 'folko', 7, '1998-02-10', '1998-03-24', '1998-02-18', 1, 88.0100021, 'folk och fä hb', 'åkergatan 24', 'bräcke', null, 's-844 67', 'sweden');
insert into orders values (10881, 'cactu', 4, '1998-02-11', '1998-03-11', '1998-02-18', 1, 2.83999991, 'cactus comidas para llevar', 'cerrito 333', 'buenos aires', null, '1010', 'argentina');
insert into orders values (10882, 'savea', 4, '1998-02-11', '1998-03-11', '1998-02-20', 3, 23.1000004, 'save-a-lot markets', '187 suffolk ln.', 'boise', 'id', '83720', 'usa');
insert into orders values (10883, 'lonep', 8, '1998-02-12', '1998-03-12', '1998-02-20', 3, 0.529999971, 'lonesome pine restaurant', '89 chiaroscuro rd.', 'portland', 'or', '97219', 'usa');
insert into orders values (10884, 'letss', 4, '1998-02-12', '1998-03-12', '1998-02-13', 2, 90.9700012, 'let''s stop n shop', '87 polk st. suite 5', 'san francisco', 'ca', '94117', 'usa');
insert into orders values (10885, 'suprd', 6, '1998-02-12', '1998-03-12', '1998-02-18', 3, 5.63999987, 'suprêmes délices', 'boulevard tirou, 255', 'charleroi', null, 'b-6000', 'belgium');
insert into orders values (10886, 'hanar', 1, '1998-02-13', '1998-03-13', '1998-03-02', 1, 4.98999977, 'hanari carnes', 'rua do paço, 67', 'rio de janeiro', 'rj', '05454-876', 'brazil');
insert into orders values (10887, 'galed', 8, '1998-02-13', '1998-03-13', '1998-02-16', 3, 1.25, 'galería del gastronómo', 'rambla de cataluña, 23', 'barcelona', null, '8022', 'spain');
insert into orders values (10888, 'godos', 1, '1998-02-16', '1998-03-16', '1998-02-23', 2, 51.8699989, 'godos cocina típica', 'c/ romero, 33', 'sevilla', null, '41101', 'spain');
insert into orders values (10889, 'rattc', 9, '1998-02-16', '1998-03-16', '1998-02-23', 3, 280.609985, 'rattlesnake canyon grocery', '2817 milton dr.', 'albuquerque', 'nm', '87110', 'usa');
insert into orders values (10890, 'dumon', 7, '1998-02-16', '1998-03-16', '1998-02-18', 1, 32.7599983, 'du monde entier', '67, rue des cinquante otages', 'nantes', null, '44000', 'france');
insert into orders values (10891, 'lehms', 7, '1998-02-17', '1998-03-17', '1998-02-19', 2, 20.3700008, 'lehmanns marktstand', 'magazinweg 7', 'frankfurt a.m.', null, '60528', 'germany');
insert into orders values (10892, 'maisd', 4, '1998-02-17', '1998-03-17', '1998-02-19', 2, 120.269997, 'maison dewey', 'rue joseph-bens 532', 'bruxelles', null, 'b-1180', 'belgium');
insert into orders values (10893, 'koene', 9, '1998-02-18', '1998-03-18', '1998-02-20', 2, 77.7799988, 'königlich essen', 'maubelstr. 90', 'brandenburg', null, '14776', 'germany');
insert into orders values (10894, 'savea', 1, '1998-02-18', '1998-03-18', '1998-02-20', 1, 116.129997, 'save-a-lot markets', '187 suffolk ln.', 'boise', 'id', '83720', 'usa');
insert into orders values (10895, 'ernsh', 3, '1998-02-18', '1998-03-18', '1998-02-23', 1, 162.75, 'ernst handel', 'kirchgasse 6', 'graz', null, '8010', 'austria');
insert into orders values (10896, 'maisd', 7, '1998-02-19', '1998-03-19', '1998-02-27', 3, 32.4500008, 'maison dewey', 'rue joseph-bens 532', 'bruxelles', null, 'b-1180', 'belgium');
insert into orders values (10897, 'hungo', 3, '1998-02-19', '1998-03-19', '1998-02-25', 2, 603.539978, 'hungry owl all-night grocers', '8 johnstown road', 'cork', 'co. cork', null, 'ireland');
insert into orders values (10898, 'ocean', 4, '1998-02-20', '1998-03-20', '1998-03-06', 2, 1.26999998, 'océano atlántico ltda.', 'ing. gustavo moncada 8585 piso 20-a', 'buenos aires', null, '1010', 'argentina');
insert into orders values (10899, 'lilas', 5, '1998-02-20', '1998-03-20', '1998-02-26', 3, 1.21000004, 'lila-supermercado', 'carrera 52 con ave. bolívar #65-98 llano largo', 'barquisimeto', 'lara', '3508', 'venezuela');
insert into orders values (10900, 'welli', 1, '1998-02-20', '1998-03-20', '1998-03-04', 2, 1.65999997, 'wellington importadora', 'rua do mercado, 12', 'resende', 'sp', '08737-363', 'brazil');
insert into orders values (10901, 'hilaa', 4, '1998-02-23', '1998-03-23', '1998-02-26', 1, 62.0900002, 'hilarion-abastos', 'carrera 22 con ave. carlos soublette #8-35', 'san cristóbal', 'táchira', '5022', 'venezuela');
insert into orders values (10902, 'folko', 1, '1998-02-23', '1998-03-23', '1998-03-03', 1, 44.1500015, 'folk och fä hb', 'åkergatan 24', 'bräcke', null, 's-844 67', 'sweden');
insert into orders values (10903, 'hanar', 3, '1998-02-24', '1998-03-24', '1998-03-04', 3, 36.7099991, 'hanari carnes', 'rua do paço, 67', 'rio de janeiro', 'rj', '05454-876', 'brazil');
insert into orders values (10904, 'whitc', 3, '1998-02-24', '1998-03-24', '1998-02-27', 3, 162.949997, 'white clover markets', '1029 - 12th ave. s.', 'seattle', 'wa', '98124', 'usa');
insert into orders values (10905, 'welli', 9, '1998-02-24', '1998-03-24', '1998-03-06', 2, 13.7200003, 'wellington importadora', 'rua do mercado, 12', 'resende', 'sp', '08737-363', 'brazil');
insert into orders values (10906, 'wolza', 4, '1998-02-25', '1998-03-11', '1998-03-03', 3, 26.2900009, 'wolski zajazd', 'ul. filtrowa 68', 'warszawa', null, '01-012', 'poland');
insert into orders values (10907, 'specd', 6, '1998-02-25', '1998-03-25', '1998-02-27', 3, 9.18999958, 'spécialités du monde', '25, rue lauriston', 'paris', null, '75016', 'france');
insert into orders values (10908, 'reggc', 4, '1998-02-26', '1998-03-26', '1998-03-06', 2, 32.9599991, 'reggiani caseifici', 'strada provinciale 124', 'reggio emilia', null, '42100', 'italy');
insert into orders values (10909, 'santg', 1, '1998-02-26', '1998-03-26', '1998-03-10', 2, 53.0499992, 'santé gourmet', 'erling skakkes gate 78', 'stavern', null, '4110', 'norway');
insert into orders values (10910, 'wilmk', 1, '1998-02-26', '1998-03-26', '1998-03-04', 3, 38.1100006, 'wilman kala', 'keskuskatu 45', 'helsinki', null, '21240', 'finland');
insert into orders values (10911, 'godos', 3, '1998-02-26', '1998-03-26', '1998-03-05', 1, 38.1899986, 'godos cocina típica', 'c/ romero, 33', 'sevilla', null, '41101', 'spain');
insert into orders values (10912, 'hungo', 2, '1998-02-26', '1998-03-26', '1998-03-18', 2, 580.909973, 'hungry owl all-night grocers', '8 johnstown road', 'cork', 'co. cork', null, 'ireland');
insert into orders values (10913, 'queen', 4, '1998-02-26', '1998-03-26', '1998-03-04', 1, 33.0499992, 'queen cozinha', 'alameda dos canàrios, 891', 'sao paulo', 'sp', '05487-020', 'brazil');
insert into orders values (10914, 'queen', 6, '1998-02-27', '1998-03-27', '1998-03-02', 1, 21.1900005, 'queen cozinha', 'alameda dos canàrios, 891', 'sao paulo', 'sp', '05487-020', 'brazil');
insert into orders values (10915, 'tortu', 2, '1998-02-27', '1998-03-27', '1998-03-02', 2, 3.50999999, 'tortuga restaurante', 'avda. azteca 123', 'méxico d.f.', null, '05033', 'mexico');
insert into orders values (10916, 'ranch', 1, '1998-02-27', '1998-03-27', '1998-03-09', 2, 63.7700005, 'rancho grande', 'av. del libertador 900', 'buenos aires', null, '1010', 'argentina');
insert into orders values (10917, 'romey', 4, '1998-03-02', '1998-03-30', '1998-03-11', 2, 8.28999996, 'romero y tomillo', 'gran vía, 1', 'madrid', null, '28001', 'spain');
insert into orders values (10918, 'bottm', 3, '1998-03-02', '1998-03-30', '1998-03-11', 3, 48.8300018, 'bottom-dollar markets', '23 tsawassen blvd.', 'tsawassen', 'bc', 't2f 8m4', 'canada');
insert into orders values (10919, 'linod', 2, '1998-03-02', '1998-03-30', '1998-03-04', 2, 19.7999992, 'lino-delicateses', 'ave. 5 de mayo porlamar', 'i. de margarita', 'nueva esparta', '4980', 'venezuela');
insert into orders values (10920, 'arout', 4, '1998-03-03', '1998-03-31', '1998-03-09', 2, 29.6100006, 'around the horn', 'brook farm stratford st. mary', 'colchester', 'essex', 'co7 6jx', 'uk');
insert into orders values (10921, 'vaffe', 1, '1998-03-03', '1998-04-14', '1998-03-09', 1, 176.479996, 'vaffeljernet', 'smagsloget 45', 'århus', null, '8200', 'denmark');
insert into orders values (10922, 'hanar', 5, '1998-03-03', '1998-03-31', '1998-03-05', 3, 62.7400017, 'hanari carnes', 'rua do paço, 67', 'rio de janeiro', 'rj', '05454-876', 'brazil');
insert into orders values (10923, 'lamai', 7, '1998-03-03', '1998-04-14', '1998-03-13', 3, 68.2600021, 'la maison d''asie', '1 rue alsace-lorraine', 'toulouse', null, '31000', 'france');
insert into orders values (10924, 'bergs', 3, '1998-03-04', '1998-04-01', '1998-04-08', 2, 151.520004, 'berglunds snabbköp', 'berguvsvägen  8', 'luleå', null, 's-958 22', 'sweden');
insert into orders values (10925, 'hanar', 3, '1998-03-04', '1998-04-01', '1998-03-13', 1, 2.26999998, 'hanari carnes', 'rua do paço, 67', 'rio de janeiro', 'rj', '05454-876', 'brazil');
insert into orders values (10926, 'anatr', 4, '1998-03-04', '1998-04-01', '1998-03-11', 3, 39.9199982, 'ana trujillo emparedados y helados', 'avda. de la constitución 2222', 'méxico d.f.', null, '05021', 'mexico');
insert into orders values (10927, 'lacor', 4, '1998-03-05', '1998-04-02', '1998-04-08', 1, 19.7900009, 'la corne d''abondance', '67, avenue de l''europe', 'versailles', null, '78000', 'france');
insert into orders values (10928, 'galed', 1, '1998-03-05', '1998-04-02', '1998-03-18', 1, 1.36000001, 'galería del gastronómo', 'rambla de cataluña, 23', 'barcelona', null, '8022', 'spain');
insert into orders values (10929, 'frank', 6, '1998-03-05', '1998-04-02', '1998-03-12', 1, 33.9300003, 'frankenversand', 'berliner platz 43', 'münchen', null, '80805', 'germany');
insert into orders values (10930, 'suprd', 4, '1998-03-06', '1998-04-17', '1998-03-18', 3, 15.5500002, 'suprêmes délices', 'boulevard tirou, 255', 'charleroi', null, 'b-6000', 'belgium');
insert into orders values (10931, 'ricsu', 4, '1998-03-06', '1998-03-20', '1998-03-19', 2, 13.6000004, 'richter supermarkt', 'starenweg 5', 'genève', null, '1204', 'switzerland');
insert into orders values (10932, 'bonap', 8, '1998-03-06', '1998-04-03', '1998-03-24', 1, 134.639999, 'bon app''', '12, rue des bouchers', 'marseille', null, '13008', 'france');
insert into orders values (10933, 'islat', 6, '1998-03-06', '1998-04-03', '1998-03-16', 3, 54.1500015, 'island trading', 'garden house crowther way', 'cowes', 'isle of wight', 'po31 7pj', 'uk');
insert into orders values (10934, 'lehms', 3, '1998-03-09', '1998-04-06', '1998-03-12', 3, 32.0099983, 'lehmanns marktstand', 'magazinweg 7', 'frankfurt a.m.', null, '60528', 'germany');
insert into orders values (10935, 'welli', 4, '1998-03-09', '1998-04-06', '1998-03-18', 3, 47.5900002, 'wellington importadora', 'rua do mercado, 12', 'resende', 'sp', '08737-363', 'brazil');
insert into orders values (10936, 'greal', 3, '1998-03-09', '1998-04-06', '1998-03-18', 2, 33.6800003, 'great lakes food market', '2732 baker blvd.', 'eugene', 'or', '97403', 'usa');
insert into orders values (10937, 'cactu', 7, '1998-03-10', '1998-03-24', '1998-03-13', 3, 31.5100002, 'cactus comidas para llevar', 'cerrito 333', 'buenos aires', null, '1010', 'argentina');
insert into orders values (10938, 'quick', 3, '1998-03-10', '1998-04-07', '1998-03-16', 2, 31.8899994, 'quick-stop', 'taucherstraße 10', 'cunewalde', null, '01307', 'germany');
insert into orders values (10939, 'magaa', 2, '1998-03-10', '1998-04-07', '1998-03-13', 2, 76.3300018, 'magazzini alimentari riuniti', 'via ludovico il moro 22', 'bergamo', null, '24100', 'italy');
insert into orders values (10940, 'bonap', 8, '1998-03-11', '1998-04-08', '1998-03-23', 3, 19.7700005, 'bon app''', '12, rue des bouchers', 'marseille', null, '13008', 'france');
insert into orders values (10941, 'savea', 7, '1998-03-11', '1998-04-08', '1998-03-20', 2, 400.809998, 'save-a-lot markets', '187 suffolk ln.', 'boise', 'id', '83720', 'usa');
insert into orders values (10942, 'reggc', 9, '1998-03-11', '1998-04-08', '1998-03-18', 3, 17.9500008, 'reggiani caseifici', 'strada provinciale 124', 'reggio emilia', null, '42100', 'italy');
insert into orders values (10943, 'bsbev', 4, '1998-03-11', '1998-04-08', '1998-03-19', 2, 2.17000008, 'b''s beverages', 'fauntleroy circus', 'london', null, 'ec2 5nt', 'uk');
insert into orders values (10944, 'bottm', 6, '1998-03-12', '1998-03-26', '1998-03-13', 3, 52.9199982, 'bottom-dollar markets', '23 tsawassen blvd.', 'tsawassen', 'bc', 't2f 8m4', 'canada');
insert into orders values (10945, 'morgk', 4, '1998-03-12', '1998-04-09', '1998-03-18', 1, 10.2200003, 'morgenstern gesundkost', 'heerstr. 22', 'leipzig', null, '04179', 'germany');
insert into orders values (10946, 'vaffe', 1, '1998-03-12', '1998-04-09', '1998-03-19', 2, 27.2000008, 'vaffeljernet', 'smagsloget 45', 'århus', null, '8200', 'denmark');
insert into orders values (10947, 'bsbev', 3, '1998-03-13', '1998-04-10', '1998-03-16', 2, 3.25999999, 'b''s beverages', 'fauntleroy circus', 'london', null, 'ec2 5nt', 'uk');
insert into orders values (10948, 'godos', 3, '1998-03-13', '1998-04-10', '1998-03-19', 3, 23.3899994, 'godos cocina típica', 'c/ romero, 33', 'sevilla', null, '41101', 'spain');
insert into orders values (10949, 'bottm', 2, '1998-03-13', '1998-04-10', '1998-03-17', 3, 74.4400024, 'bottom-dollar markets', '23 tsawassen blvd.', 'tsawassen', 'bc', 't2f 8m4', 'canada');
insert into orders values (10950, 'magaa', 1, '1998-03-16', '1998-04-13', '1998-03-23', 2, 2.5, 'magazzini alimentari riuniti', 'via ludovico il moro 22', 'bergamo', null, '24100', 'italy');
insert into orders values (10951, 'ricsu', 9, '1998-03-16', '1998-04-27', '1998-04-07', 2, 30.8500004, 'richter supermarkt', 'starenweg 5', 'genève', null, '1204', 'switzerland');
insert into orders values (10952, 'alfki', 1, '1998-03-16', '1998-04-27', '1998-03-24', 1, 40.4199982, 'alfred''s futterkiste', 'obere str. 57', 'berlin', null, '12209', 'germany');
insert into orders values (10953, 'arout', 9, '1998-03-16', '1998-03-30', '1998-03-25', 2, 23.7199993, 'around the horn', 'brook farm stratford st. mary', 'colchester', 'essex', 'co7 6jx', 'uk');
insert into orders values (10954, 'linod', 5, '1998-03-17', '1998-04-28', '1998-03-20', 1, 27.9099998, 'lino-delicateses', 'ave. 5 de mayo porlamar', 'i. de margarita', 'nueva esparta', '4980', 'venezuela');
insert into orders values (10955, 'folko', 8, '1998-03-17', '1998-04-14', '1998-03-20', 2, 3.25999999, 'folk och fä hb', 'åkergatan 24', 'bräcke', null, 's-844 67', 'sweden');
insert into orders values (10956, 'blaus', 6, '1998-03-17', '1998-04-28', '1998-03-20', 2, 44.6500015, 'blauer see delikatessen', 'forsterstr. 57', 'mannheim', null, '68306', 'germany');
insert into orders values (10957, 'hilaa', 8, '1998-03-18', '1998-04-15', '1998-03-27', 3, 105.360001, 'hilarion-abastos', 'carrera 22 con ave. carlos soublette #8-35', 'san cristóbal', 'táchira', '5022', 'venezuela');
insert into orders values (10958, 'ocean', 7, '1998-03-18', '1998-04-15', '1998-03-27', 2, 49.5600014, 'océano atlántico ltda.', 'ing. gustavo moncada 8585 piso 20-a', 'buenos aires', null, '1010', 'argentina');
insert into orders values (10959, 'gourl', 6, '1998-03-18', '1998-04-29', '1998-03-23', 2, 4.98000002, 'gourmet lanchonetes', 'av. brasil, 442', 'campinas', 'sp', '04876-786', 'brazil');
insert into orders values (10960, 'hilaa', 3, '1998-03-19', '1998-04-02', '1998-04-08', 1, 2.07999992, 'hilarion-abastos', 'carrera 22 con ave. carlos soublette #8-35', 'san cristóbal', 'táchira', '5022', 'venezuela');
insert into orders values (10961, 'queen', 8, '1998-03-19', '1998-04-16', '1998-03-30', 1, 104.470001, 'queen cozinha', 'alameda dos canàrios, 891', 'sao paulo', 'sp', '05487-020', 'brazil');
insert into orders values (10962, 'quick', 8, '1998-03-19', '1998-04-16', '1998-03-23', 2, 275.790009, 'quick-stop', 'taucherstraße 10', 'cunewalde', null, '01307', 'germany');
insert into orders values (10963, 'furib', 9, '1998-03-19', '1998-04-16', '1998-03-26', 3, 2.70000005, 'furia bacalhau e frutos do mar', 'jardim das rosas n. 32', 'lisboa', null, '1675', 'portugal');
insert into orders values (10964, 'specd', 3, '1998-03-20', '1998-04-17', '1998-03-24', 2, 87.3799973, 'spécialités du monde', '25, rue lauriston', 'paris', null, '75016', 'france');
insert into orders values (10965, 'oldwo', 6, '1998-03-20', '1998-04-17', '1998-03-30', 3, 144.380005, 'old world delicatessen', '2743 bering st.', 'anchorage', 'ak', '99508', 'usa');
insert into orders values (10966, 'chops', 4, '1998-03-20', '1998-04-17', '1998-04-08', 1, 27.1900005, 'chop-suey chinese', 'hauptstr. 31', 'bern', null, '3012', 'switzerland');
insert into orders values (10967, 'tomsp', 2, '1998-03-23', '1998-04-20', '1998-04-02', 2, 62.2200012, 'toms spezialitäten', 'luisenstr. 48', 'münster', null, '44087', 'germany');
insert into orders values (10968, 'ernsh', 1, '1998-03-23', '1998-04-20', '1998-04-01', 3, 74.5999985, 'ernst handel', 'kirchgasse 6', 'graz', null, '8010', 'austria');
insert into orders values (10969, 'commi', 1, '1998-03-23', '1998-04-20', '1998-03-30', 2, 0.209999993, 'comércio mineiro', 'av. dos lusíadas, 23', 'sao paulo', 'sp', '05432-043', 'brazil');
insert into orders values (10970, 'bolid', 9, '1998-03-24', '1998-04-07', '1998-04-24', 1, 16.1599998, 'bólido comidas preparadas', 'c/ araquil, 67', 'madrid', null, '28023', 'spain');
insert into orders values (10971, 'franr', 2, '1998-03-24', '1998-04-21', '1998-04-02', 2, 121.82, 'france restauration', '54, rue royale', 'nantes', null, '44000', 'france');
insert into orders values (10972, 'lacor', 4, '1998-03-24', '1998-04-21', '1998-03-26', 2, 0.0199999996, 'la corne d''abondance', '67, avenue de l''europe', 'versailles', null, '78000', 'france');
insert into orders values (10973, 'lacor', 6, '1998-03-24', '1998-04-21', '1998-03-27', 2, 15.1700001, 'la corne d''abondance', '67, avenue de l''europe', 'versailles', null, '78000', 'france');
insert into orders values (10974, 'splir', 3, '1998-03-25', '1998-04-08', '1998-04-03', 3, 12.96, 'split rail beer & ale', 'p.o. box 555', 'lander', 'wy', '82520', 'usa');
insert into orders values (10975, 'bottm', 1, '1998-03-25', '1998-04-22', '1998-03-27', 3, 32.2700005, 'bottom-dollar markets', '23 tsawassen blvd.', 'tsawassen', 'bc', 't2f 8m4', 'canada');
insert into orders values (10976, 'hilaa', 1, '1998-03-25', '1998-05-06', '1998-04-03', 1, 37.9700012, 'hilarion-abastos', 'carrera 22 con ave. carlos soublette #8-35', 'san cristóbal', 'táchira', '5022', 'venezuela');
insert into orders values (10977, 'folko', 8, '1998-03-26', '1998-04-23', '1998-04-10', 3, 208.5, 'folk och fä hb', 'åkergatan 24', 'bräcke', null, 's-844 67', 'sweden');
insert into orders values (10978, 'maisd', 9, '1998-03-26', '1998-04-23', '1998-04-23', 2, 32.8199997, 'maison dewey', 'rue joseph-bens 532', 'bruxelles', null, 'b-1180', 'belgium');
insert into orders values (10979, 'ernsh', 8, '1998-03-26', '1998-04-23', '1998-03-31', 2, 353.070007, 'ernst handel', 'kirchgasse 6', 'graz', null, '8010', 'austria');
insert into orders values (10980, 'folko', 4, '1998-03-27', '1998-05-08', '1998-04-17', 1, 1.25999999, 'folk och fä hb', 'åkergatan 24', 'bräcke', null, 's-844 67', 'sweden');
insert into orders values (10981, 'hanar', 1, '1998-03-27', '1998-04-24', '1998-04-02', 2, 193.369995, 'hanari carnes', 'rua do paço, 67', 'rio de janeiro', 'rj', '05454-876', 'brazil');
insert into orders values (10982, 'bottm', 2, '1998-03-27', '1998-04-24', '1998-04-08', 1, 14.0100002, 'bottom-dollar markets', '23 tsawassen blvd.', 'tsawassen', 'bc', 't2f 8m4', 'canada');
insert into orders values (10983, 'savea', 2, '1998-03-27', '1998-04-24', '1998-04-06', 2, 657.539978, 'save-a-lot markets', '187 suffolk ln.', 'boise', 'id', '83720', 'usa');
insert into orders values (10984, 'savea', 1, '1998-03-30', '1998-04-27', '1998-04-03', 3, 211.220001, 'save-a-lot markets', '187 suffolk ln.', 'boise', 'id', '83720', 'usa');
insert into orders values (10985, 'hungo', 2, '1998-03-30', '1998-04-27', '1998-04-02', 1, 91.5100021, 'hungry owl all-night grocers', '8 johnstown road', 'cork', 'co. cork', null, 'ireland');
insert into orders values (10986, 'ocean', 8, '1998-03-30', '1998-04-27', '1998-04-21', 2, 217.860001, 'océano atlántico ltda.', 'ing. gustavo moncada 8585 piso 20-a', 'buenos aires', null, '1010', 'argentina');
insert into orders values (10987, 'eastc', 8, '1998-03-31', '1998-04-28', '1998-04-06', 1, 185.479996, 'eastern connection', '35 king george', 'london', null, 'wx3 6fw', 'uk');
insert into orders values (10988, 'rattc', 3, '1998-03-31', '1998-04-28', '1998-04-10', 2, 61.1399994, 'rattlesnake canyon grocery', '2817 milton dr.', 'albuquerque', 'nm', '87110', 'usa');
insert into orders values (10989, 'quede', 2, '1998-03-31', '1998-04-28', '1998-04-02', 1, 34.7599983, 'que delícia', 'rua da panificadora, 12', 'rio de janeiro', 'rj', '02389-673', 'brazil');
insert into orders values (10990, 'ernsh', 2, '1998-04-01', '1998-05-13', '1998-04-07', 3, 117.610001, 'ernst handel', 'kirchgasse 6', 'graz', null, '8010', 'austria');
insert into orders values (10991, 'quick', 1, '1998-04-01', '1998-04-29', '1998-04-07', 1, 38.5099983, 'quick-stop', 'taucherstraße 10', 'cunewalde', null, '01307', 'germany');
insert into orders values (10992, 'thebi', 1, '1998-04-01', '1998-04-29', '1998-04-03', 3, 4.26999998, 'the big cheese', '89 jefferson way suite 2', 'portland', 'or', '97201', 'usa');
insert into orders values (10993, 'folko', 7, '1998-04-01', '1998-04-29', '1998-04-10', 3, 8.81000042, 'folk och fä hb', 'åkergatan 24', 'bräcke', null, 's-844 67', 'sweden');
insert into orders values (10994, 'vaffe', 2, '1998-04-02', '1998-04-16', '1998-04-09', 3, 65.5299988, 'vaffeljernet', 'smagsloget 45', 'århus', null, '8200', 'denmark');
insert into orders values (10995, 'peric', 1, '1998-04-02', '1998-04-30', '1998-04-06', 3, 46, 'pericles comidas clásicas', 'calle dr. jorge cash 321', 'méxico d.f.', null, '05033', 'mexico');
insert into orders values (10996, 'quick', 4, '1998-04-02', '1998-04-30', '1998-04-10', 2, 1.12, 'quick-stop', 'taucherstraße 10', 'cunewalde', null, '01307', 'germany');
insert into orders values (10997, 'lilas', 8, '1998-04-03', '1998-05-15', '1998-04-13', 2, 73.9100037, 'lila-supermercado', 'carrera 52 con ave. bolívar #65-98 llano largo', 'barquisimeto', 'lara', '3508', 'venezuela');
insert into orders values (10998, 'wolza', 8, '1998-04-03', '1998-04-17', '1998-04-17', 2, 20.3099995, 'wolski zajazd', 'ul. filtrowa 68', 'warszawa', null, '01-012', 'poland');
insert into orders values (10999, 'ottik', 6, '1998-04-03', '1998-05-01', '1998-04-10', 2, 96.3499985, 'ottilies käseladen', 'mehrheimerstr. 369', 'köln', null, '50739', 'germany');
insert into orders values (11000, 'rattc', 2, '1998-04-06', '1998-05-04', '1998-04-14', 3, 55.1199989, 'rattlesnake canyon grocery', '2817 milton dr.', 'albuquerque', 'nm', '87110', 'usa');
insert into orders values (11001, 'folko', 2, '1998-04-06', '1998-05-04', '1998-04-14', 2, 197.300003, 'folk och fä hb', 'åkergatan 24', 'bräcke', null, 's-844 67', 'sweden');
insert into orders values (11002, 'savea', 4, '1998-04-06', '1998-05-04', '1998-04-16', 1, 141.160004, 'save-a-lot markets', '187 suffolk ln.', 'boise', 'id', '83720', 'usa');
insert into orders values (11003, 'thecr', 3, '1998-04-06', '1998-05-04', '1998-04-08', 3, 14.9099998, 'the cracker box', '55 grizzly peak rd.', 'butte', 'mt', '59801', 'usa');
insert into orders values (11004, 'maisd', 3, '1998-04-07', '1998-05-05', '1998-04-20', 1, 44.8400002, 'maison dewey', 'rue joseph-bens 532', 'bruxelles', null, 'b-1180', 'belgium');
insert into orders values (11005, 'wilmk', 2, '1998-04-07', '1998-05-05', '1998-04-10', 1, 0.75, 'wilman kala', 'keskuskatu 45', 'helsinki', null, '21240', 'finland');
insert into orders values (11006, 'greal', 3, '1998-04-07', '1998-05-05', '1998-04-15', 2, 25.1900005, 'great lakes food market', '2732 baker blvd.', 'eugene', 'or', '97403', 'usa');
insert into orders values (11007, 'prini', 8, '1998-04-08', '1998-05-06', '1998-04-13', 2, 202.240005, 'princesa isabel vinhos', 'estrada da saúde n. 58', 'lisboa', null, '1756', 'portugal');
insert into orders values (11008, 'ernsh', 7, '1998-04-08', '1998-05-06', null, 3, 79.4599991, 'ernst handel', 'kirchgasse 6', 'graz', null, '8010', 'austria');
insert into orders values (11009, 'godos', 2, '1998-04-08', '1998-05-06', '1998-04-10', 1, 59.1100006, 'godos cocina típica', 'c/ romero, 33', 'sevilla', null, '41101', 'spain');
insert into orders values (11010, 'reggc', 2, '1998-04-09', '1998-05-07', '1998-04-21', 2, 28.7099991, 'reggiani caseifici', 'strada provinciale 124', 'reggio emilia', null, '42100', 'italy');
insert into orders values (11011, 'alfki', 3, '1998-04-09', '1998-05-07', '1998-04-13', 1, 1.21000004, 'alfred''s futterkiste', 'obere str. 57', 'berlin', null, '12209', 'germany');
insert into orders values (11012, 'frank', 1, '1998-04-09', '1998-04-23', '1998-04-17', 3, 242.949997, 'frankenversand', 'berliner platz 43', 'münchen', null, '80805', 'germany');
insert into orders values (11013, 'romey', 2, '1998-04-09', '1998-05-07', '1998-04-10', 1, 32.9900017, 'romero y tomillo', 'gran vía, 1', 'madrid', null, '28001', 'spain');
insert into orders values (11014, 'linod', 2, '1998-04-10', '1998-05-08', '1998-04-15', 3, 23.6000004, 'lino-delicateses', 'ave. 5 de mayo porlamar', 'i. de margarita', 'nueva esparta', '4980', 'venezuela');
insert into orders values (11015, 'santg', 2, '1998-04-10', '1998-04-24', '1998-04-20', 2, 4.61999989, 'santé gourmet', 'erling skakkes gate 78', 'stavern', null, '4110', 'norway');
insert into orders values (11016, 'arout', 9, '1998-04-10', '1998-05-08', '1998-04-13', 2, 33.7999992, 'around the horn', 'brook farm stratford st. mary', 'colchester', 'essex', 'co7 6jx', 'uk');
insert into orders values (11017, 'ernsh', 9, '1998-04-13', '1998-05-11', '1998-04-20', 2, 754.26001, 'ernst handel', 'kirchgasse 6', 'graz', null, '8010', 'austria');
insert into orders values (11018, 'lonep', 4, '1998-04-13', '1998-05-11', '1998-04-16', 2, 11.6499996, 'lonesome pine restaurant', '89 chiaroscuro rd.', 'portland', 'or', '97219', 'usa');
insert into orders values (11019, 'ranch', 6, '1998-04-13', '1998-05-11', null, 3, 3.17000008, 'rancho grande', 'av. del libertador 900', 'buenos aires', null, '1010', 'argentina');
insert into orders values (11020, 'ottik', 2, '1998-04-14', '1998-05-12', '1998-04-16', 2, 43.2999992, 'ottilies käseladen', 'mehrheimerstr. 369', 'köln', null, '50739', 'germany');
insert into orders values (11021, 'quick', 3, '1998-04-14', '1998-05-12', '1998-04-21', 1, 297.179993, 'quick-stop', 'taucherstraße 10', 'cunewalde', null, '01307', 'germany');
insert into orders values (11022, 'hanar', 9, '1998-04-14', '1998-05-12', '1998-05-04', 2, 6.26999998, 'hanari carnes', 'rua do paço, 67', 'rio de janeiro', 'rj', '05454-876', 'brazil');
insert into orders values (11023, 'bsbev', 1, '1998-04-14', '1998-04-28', '1998-04-24', 2, 123.830002, 'b''s beverages', 'fauntleroy circus', 'london', null, 'ec2 5nt', 'uk');
insert into orders values (11024, 'eastc', 4, '1998-04-15', '1998-05-13', '1998-04-20', 1, 74.3600006, 'eastern connection', '35 king george', 'london', null, 'wx3 6fw', 'uk');
insert into orders values (11025, 'warth', 6, '1998-04-15', '1998-05-13', '1998-04-24', 3, 29.1700001, 'wartian herkku', 'torikatu 38', 'oulu', null, '90110', 'finland');
insert into orders values (11026, 'frans', 4, '1998-04-15', '1998-05-13', '1998-04-28', 1, 47.0900002, 'franchi s.p.a.', 'via monte bianco 34', 'torino', null, '10100', 'italy');
insert into orders values (11027, 'bottm', 1, '1998-04-16', '1998-05-14', '1998-04-20', 1, 52.5200005, 'bottom-dollar markets', '23 tsawassen blvd.', 'tsawassen', 'bc', 't2f 8m4', 'canada');
insert into orders values (11028, 'koene', 2, '1998-04-16', '1998-05-14', '1998-04-22', 1, 29.5900002, 'königlich essen', 'maubelstr. 90', 'brandenburg', null, '14776', 'germany');
insert into orders values (11029, 'chops', 4, '1998-04-16', '1998-05-14', '1998-04-27', 1, 47.8400002, 'chop-suey chinese', 'hauptstr. 31', 'bern', null, '3012', 'switzerland');
insert into orders values (11030, 'savea', 7, '1998-04-17', '1998-05-15', '1998-04-27', 2, 830.75, 'save-a-lot markets', '187 suffolk ln.', 'boise', 'id', '83720', 'usa');
insert into orders values (11031, 'savea', 6, '1998-04-17', '1998-05-15', '1998-04-24', 2, 227.220001, 'save-a-lot markets', '187 suffolk ln.', 'boise', 'id', '83720', 'usa');
insert into orders values (11032, 'whitc', 2, '1998-04-17', '1998-05-15', '1998-04-23', 3, 606.190002, 'white clover markets', '1029 - 12th ave. s.', 'seattle', 'wa', '98124', 'usa');
insert into orders values (11033, 'ricsu', 7, '1998-04-17', '1998-05-15', '1998-04-23', 3, 84.7399979, 'richter supermarkt', 'starenweg 5', 'genève', null, '1204', 'switzerland');
insert into orders values (11034, 'oldwo', 8, '1998-04-20', '1998-06-01', '1998-04-27', 1, 40.3199997, 'old world delicatessen', '2743 bering st.', 'anchorage', 'ak', '99508', 'usa');
insert into orders values (11035, 'suprd', 2, '1998-04-20', '1998-05-18', '1998-04-24', 2, 0.170000002, 'suprêmes délices', 'boulevard tirou, 255', 'charleroi', null, 'b-6000', 'belgium');
insert into orders values (11036, 'dracd', 8, '1998-04-20', '1998-05-18', '1998-04-22', 3, 149.470001, 'drachenblut delikatessen', 'walserweg 21', 'aachen', null, '52066', 'germany');
insert into orders values (11037, 'godos', 7, '1998-04-21', '1998-05-19', '1998-04-27', 1, 3.20000005, 'godos cocina típica', 'c/ romero, 33', 'sevilla', null, '41101', 'spain');
insert into orders values (11038, 'suprd', 1, '1998-04-21', '1998-05-19', '1998-04-30', 2, 29.5900002, 'suprêmes délices', 'boulevard tirou, 255', 'charleroi', null, 'b-6000', 'belgium');
insert into orders values (11039, 'linod', 1, '1998-04-21', '1998-05-19', null, 2, 65, 'lino-delicateses', 'ave. 5 de mayo porlamar', 'i. de margarita', 'nueva esparta', '4980', 'venezuela');
insert into orders values (11040, 'greal', 4, '1998-04-22', '1998-05-20', null, 3, 18.8400002, 'great lakes food market', '2732 baker blvd.', 'eugene', 'or', '97403', 'usa');
insert into orders values (11041, 'chops', 3, '1998-04-22', '1998-05-20', '1998-04-28', 2, 48.2200012, 'chop-suey chinese', 'hauptstr. 31', 'bern', null, '3012', 'switzerland');
insert into orders values (11042, 'commi', 2, '1998-04-22', '1998-05-06', '1998-05-01', 1, 29.9899998, 'comércio mineiro', 'av. dos lusíadas, 23', 'sao paulo', 'sp', '05432-043', 'brazil');
insert into orders values (11043, 'specd', 5, '1998-04-22', '1998-05-20', '1998-04-29', 2, 8.80000019, 'spécialités du monde', '25, rue lauriston', 'paris', null, '75016', 'france');
insert into orders values (11044, 'wolza', 4, '1998-04-23', '1998-05-21', '1998-05-01', 1, 8.72000027, 'wolski zajazd', 'ul. filtrowa 68', 'warszawa', null, '01-012', 'poland');
insert into orders values (11045, 'bottm', 6, '1998-04-23', '1998-05-21', null, 2, 70.5800018, 'bottom-dollar markets', '23 tsawassen blvd.', 'tsawassen', 'bc', 't2f 8m4', 'canada');
insert into orders values (11046, 'wandk', 8, '1998-04-23', '1998-05-21', '1998-04-24', 2, 71.6399994, 'die wandernde kuh', 'adenauerallee 900', 'stuttgart', null, '70563', 'germany');
insert into orders values (11047, 'eastc', 7, '1998-04-24', '1998-05-22', '1998-05-01', 3, 46.6199989, 'eastern connection', '35 king george', 'london', null, 'wx3 6fw', 'uk');
insert into orders values (11048, 'bottm', 7, '1998-04-24', '1998-05-22', '1998-04-30', 3, 24.1200008, 'bottom-dollar markets', '23 tsawassen blvd.', 'tsawassen', 'bc', 't2f 8m4', 'canada');
insert into orders values (11049, 'gourl', 3, '1998-04-24', '1998-05-22', '1998-05-04', 1, 8.34000015, 'gourmet lanchonetes', 'av. brasil, 442', 'campinas', 'sp', '04876-786', 'brazil');
insert into orders values (11050, 'folko', 8, '1998-04-27', '1998-05-25', '1998-05-05', 2, 59.4099998, 'folk och fä hb', 'åkergatan 24', 'bräcke', null, 's-844 67', 'sweden');
insert into orders values (11051, 'lamai', 7, '1998-04-27', '1998-05-25', null, 3, 2.78999996, 'la maison d''asie', '1 rue alsace-lorraine', 'toulouse', null, '31000', 'france');
insert into orders values (11052, 'hanar', 3, '1998-04-27', '1998-05-25', '1998-05-01', 1, 67.2600021, 'hanari carnes', 'rua do paço, 67', 'rio de janeiro', 'rj', '05454-876', 'brazil');
insert into orders values (11053, 'picco', 2, '1998-04-27', '1998-05-25', '1998-04-29', 2, 53.0499992, 'piccolo und mehr', 'geislweg 14', 'salzburg', null, '5020', 'austria');
insert into orders values (11054, 'cactu', 8, '1998-04-28', '1998-05-26', null, 1, 0.330000013, 'cactus comidas para llevar', 'cerrito 333', 'buenos aires', null, '1010', 'argentina');
insert into orders values (11055, 'hilaa', 7, '1998-04-28', '1998-05-26', '1998-05-05', 2, 120.919998, 'hilarion-abastos', 'carrera 22 con ave. carlos soublette #8-35', 'san cristóbal', 'táchira', '5022', 'venezuela');
insert into orders values (11056, 'eastc', 8, '1998-04-28', '1998-05-12', '1998-05-01', 2, 278.959991, 'eastern connection', '35 king george', 'london', null, 'wx3 6fw', 'uk');
insert into orders values (11057, 'norts', 3, '1998-04-29', '1998-05-27', '1998-05-01', 3, 4.13000011, 'north/south', 'south house 300 queensbridge', 'london', null, 'sw7 1rz', 'uk');
insert into orders values (11058, 'blaus', 9, '1998-04-29', '1998-05-27', null, 3, 31.1399994, 'blauer see delikatessen', 'forsterstr. 57', 'mannheim', null, '68306', 'germany');
insert into orders values (11059, 'ricar', 2, '1998-04-29', '1998-06-10', null, 2, 85.8000031, 'ricardo adocicados', 'av. copacabana, 267', 'rio de janeiro', 'rj', '02389-890', 'brazil');
insert into orders values (11060, 'frans', 2, '1998-04-30', '1998-05-28', '1998-05-04', 2, 10.9799995, 'franchi s.p.a.', 'via monte bianco 34', 'torino', null, '10100', 'italy');
insert into orders values (11061, 'greal', 4, '1998-04-30', '1998-06-11', null, 3, 14.0100002, 'great lakes food market', '2732 baker blvd.', 'eugene', 'or', '97403', 'usa');
insert into orders values (11062, 'reggc', 4, '1998-04-30', '1998-05-28', null, 2, 29.9300003, 'reggiani caseifici', 'strada provinciale 124', 'reggio emilia', null, '42100', 'italy');
insert into orders values (11063, 'hungo', 3, '1998-04-30', '1998-05-28', '1998-05-06', 2, 81.7300034, 'hungry owl all-night grocers', '8 johnstown road', 'cork', 'co. cork', null, 'ireland');
insert into orders values (11064, 'savea', 1, '1998-05-01', '1998-05-29', '1998-05-04', 1, 30.0900002, 'save-a-lot markets', '187 suffolk ln.', 'boise', 'id', '83720', 'usa');
insert into orders values (11065, 'lilas', 8, '1998-05-01', '1998-05-29', null, 1, 12.9099998, 'lila-supermercado', 'carrera 52 con ave. bolívar #65-98 llano largo', 'barquisimeto', 'lara', '3508', 'venezuela');
insert into orders values (11066, 'whitc', 7, '1998-05-01', '1998-05-29', '1998-05-04', 2, 44.7200012, 'white clover markets', '1029 - 12th ave. s.', 'seattle', 'wa', '98124', 'usa');
insert into orders values (11067, 'dracd', 1, '1998-05-04', '1998-05-18', '1998-05-06', 2, 7.98000002, 'drachenblut delikatessen', 'walserweg 21', 'aachen', null, '52066', 'germany');
insert into orders values (11068, 'queen', 8, '1998-05-04', '1998-06-01', null, 2, 81.75, 'queen cozinha', 'alameda dos canàrios, 891', 'sao paulo', 'sp', '05487-020', 'brazil');
insert into orders values (11069, 'tortu', 1, '1998-05-04', '1998-06-01', '1998-05-06', 2, 15.6700001, 'tortuga restaurante', 'avda. azteca 123', 'méxico d.f.', null, '05033', 'mexico');
insert into orders values (11070, 'lehms', 2, '1998-05-05', '1998-06-02', null, 1, 136, 'lehmanns marktstand', 'magazinweg 7', 'frankfurt a.m.', null, '60528', 'germany');
insert into orders values (11071, 'lilas', 1, '1998-05-05', '1998-06-02', null, 1, 0.930000007, 'lila-supermercado', 'carrera 52 con ave. bolívar #65-98 llano largo', 'barquisimeto', 'lara', '3508', 'venezuela');
insert into orders values (11072, 'ernsh', 4, '1998-05-05', '1998-06-02', null, 2, 258.640015, 'ernst handel', 'kirchgasse 6', 'graz', null, '8010', 'austria');
insert into orders values (11073, 'peric', 2, '1998-05-05', '1998-06-02', null, 2, 24.9500008, 'pericles comidas clásicas', 'calle dr. jorge cash 321', 'méxico d.f.', null, '05033', 'mexico');
insert into orders values (11074, 'simob', 7, '1998-05-06', '1998-06-03', null, 2, 18.4400005, 'simons bistro', 'vinbæltet 34', 'kobenhavn', null, '1734', 'denmark');
insert into orders values (11075, 'ricsu', 8, '1998-05-06', '1998-06-03', null, 2, 6.19000006, 'richter supermarkt', 'starenweg 5', 'genève', null, '1204', 'switzerland');
insert into orders values (11076, 'bonap', 4, '1998-05-06', '1998-06-03', null, 2, 38.2799988, 'bon app''', '12, rue des bouchers', 'marseille', null, '13008', 'france');
insert into orders values (11077, 'rattc', 1, '1998-05-06', '1998-06-03', null, 2, 8.52999973, 'rattlesnake canyon grocery', '2817 milton dr.', 'albuquerque', 'nm', '87110', 'usa');


--
-- data for name: products; type: table data; schema: public; owner: postgres
--

insert into products values (1, 'chai', 8, 1, '10 boxes x 30 bags', 18, 39, 0, 10, 1);
insert into products values (2, 'chang', 1, 1, '24 - 12 oz bottles', 19, 17, 40, 25, 1);
insert into products values (3, 'aniseed syrup', 1, 2, '12 - 550 ml bottles', 10, 13, 70, 25, 0);
insert into products values (4, 'chef anton''s cajun seasoning', 2, 2, '48 - 6 oz jars', 22, 53, 0, 0, 0);
insert into products values (5, 'chef anton''s gumbo mix', 2, 2, '36 boxes', 21.3500004, 0, 0, 0, 1);
insert into products values (6, 'grandma''s boysenberry spread', 3, 2, '12 - 8 oz jars', 25, 120, 0, 25, 0);
insert into products values (7, 'uncle bob''s organic dried pears', 3, 7, '12 - 1 lb pkgs.', 30, 15, 0, 10, 0);
insert into products values (8, 'northwoods cranberry sauce', 3, 2, '12 - 12 oz jars', 40, 6, 0, 0, 0);
insert into products values (9, 'mishi kobe niku', 4, 6, '18 - 500 g pkgs.', 97, 29, 0, 0, 1);
insert into products values (10, 'ikura', 4, 8, '12 - 200 ml jars', 31, 31, 0, 0, 0);
insert into products values (11, 'queso cabrales', 5, 4, '1 kg pkg.', 21, 22, 30, 30, 0);
insert into products values (12, 'queso manchego la pastora', 5, 4, '10 - 500 g pkgs.', 38, 86, 0, 0, 0);
insert into products values (13, 'konbu', 6, 8, '2 kg box', 6, 24, 0, 5, 0);
insert into products values (14, 'tofu', 6, 7, '40 - 100 g pkgs.', 23.25, 35, 0, 0, 0);
insert into products values (15, 'genen shouyu', 6, 2, '24 - 250 ml bottles', 13, 39, 0, 5, 0);
insert into products values (16, 'pavlova', 7, 3, '32 - 500 g boxes', 17.4500008, 29, 0, 10, 0);
insert into products values (17, 'alice mutton', 7, 6, '20 - 1 kg tins', 39, 0, 0, 0, 1);
insert into products values (18, 'carnarvon tigers', 7, 8, '16 kg pkg.', 62.5, 42, 0, 0, 0);
insert into products values (19, 'teatime chocolate biscuits', 8, 3, '10 boxes x 12 pieces', 9.19999981, 25, 0, 5, 0);
insert into products values (20, 'sir rodney''s marmalade', 8, 3, '30 gift boxes', 81, 40, 0, 0, 0);
insert into products values (21, 'sir rodney''s scones', 8, 3, '24 pkgs. x 4 pieces', 10, 3, 40, 5, 0);
insert into products values (22, 'gustaf''s knäckebröd', 9, 5, '24 - 500 g pkgs.', 21, 104, 0, 25, 0);
insert into products values (23, 'tunnbröd', 9, 5, '12 - 250 g pkgs.', 9, 61, 0, 25, 0);
insert into products values (24, 'guaraná fantástica', 10, 1, '12 - 355 ml cans', 4.5, 20, 0, 0, 1);
insert into products values (25, 'nunuca nuß-nougat-creme', 11, 3, '20 - 450 g glasses', 14, 76, 0, 30, 0);
insert into products values (26, 'gumbär gummibärchen', 11, 3, '100 - 250 g bags', 31.2299995, 15, 0, 0, 0);
insert into products values (27, 'schoggi schokolade', 11, 3, '100 - 100 g pieces', 43.9000015, 49, 0, 30, 0);
insert into products values (28, 'rössle sauerkraut', 12, 7, '25 - 825 g cans', 45.5999985, 26, 0, 0, 1);
insert into products values (29, 'thüringer rostbratwurst', 12, 6, '50 bags x 30 sausgs.', 123.790001, 0, 0, 0, 1);
insert into products values (30, 'nord-ost matjeshering', 13, 8, '10 - 200 g glasses', 25.8899994, 10, 0, 15, 0);
insert into products values (31, 'gorgonzola telino', 14, 4, '12 - 100 g pkgs', 12.5, 0, 70, 20, 0);
insert into products values (32, 'mascarpone fabioli', 14, 4, '24 - 200 g pkgs.', 32, 9, 40, 25, 0);
insert into products values (33, 'geitost', 15, 4, '500 g', 2.5, 112, 0, 20, 0);
insert into products values (34, 'sasquatch ale', 16, 1, '24 - 12 oz bottles', 14, 111, 0, 15, 0);
insert into products values (35, 'steeleye stout', 16, 1, '24 - 12 oz bottles', 18, 20, 0, 15, 0);
insert into products values (36, 'inlagd sill', 17, 8, '24 - 250 g  jars', 19, 112, 0, 20, 0);
insert into products values (37, 'gravad lax', 17, 8, '12 - 500 g pkgs.', 26, 11, 50, 25, 0);
insert into products values (38, 'côte de blaye', 18, 1, '12 - 75 cl bottles', 263.5, 17, 0, 15, 0);
insert into products values (39, 'chartreuse verte', 18, 1, '750 cc per bottle', 18, 69, 0, 5, 0);
insert into products values (40, 'boston crab meat', 19, 8, '24 - 4 oz tins', 18.3999996, 123, 0, 30, 0);
insert into products values (41, 'jack''s new england clam chowder', 19, 8, '12 - 12 oz cans', 9.64999962, 85, 0, 10, 0);
insert into products values (42, 'singaporean hokkien fried mee', 20, 5, '32 - 1 kg pkgs.', 14, 26, 0, 0, 1);
insert into products values (43, 'ipoh coffee', 20, 1, '16 - 500 g tins', 46, 17, 10, 25, 0);
insert into products values (44, 'gula malacca', 20, 2, '20 - 2 kg bags', 19.4500008, 27, 0, 15, 0);
insert into products values (45, 'rogede sild', 21, 8, '1k pkg.', 9.5, 5, 70, 15, 0);
insert into products values (46, 'spegesild', 21, 8, '4 - 450 g glasses', 12, 95, 0, 0, 0);
insert into products values (47, 'zaanse koeken', 22, 3, '10 - 4 oz boxes', 9.5, 36, 0, 0, 0);
insert into products values (48, 'chocolade', 22, 3, '10 pkgs.', 12.75, 15, 70, 25, 0);
insert into products values (49, 'maxilaku', 23, 3, '24 - 50 g pkgs.', 20, 10, 60, 15, 0);
insert into products values (50, 'valkoinen suklaa', 23, 3, '12 - 100 g bars', 16.25, 65, 0, 30, 0);
insert into products values (51, 'manjimup dried apples', 24, 7, '50 - 300 g pkgs.', 53, 20, 0, 10, 0);
insert into products values (52, 'filo mix', 24, 5, '16 - 2 kg boxes', 7, 38, 0, 25, 0);
insert into products values (53, 'perth pasties', 24, 6, '48 pieces', 32.7999992, 0, 0, 0, 1);
insert into products values (54, 'tourtière', 25, 6, '16 pies', 7.44999981, 21, 0, 10, 0);
insert into products values (55, 'pâté chinois', 25, 6, '24 boxes x 2 pies', 24, 115, 0, 20, 0);
insert into products values (56, 'gnocchi di nonna alice', 26, 5, '24 - 250 g pkgs.', 38, 21, 10, 30, 0);
insert into products values (57, 'ravioli angelo', 26, 5, '24 - 250 g pkgs.', 19.5, 36, 0, 20, 0);
insert into products values (58, 'escargots de bourgogne', 27, 8, '24 pieces', 13.25, 62, 0, 20, 0);
insert into products values (59, 'raclette courdavault', 28, 4, '5 kg pkg.', 55, 79, 0, 0, 0);
insert into products values (60, 'camembert pierrot', 28, 4, '15 - 300 g rounds', 34, 19, 0, 0, 0);
insert into products values (61, 'sirop d''érable', 29, 2, '24 - 500 ml bottles', 28.5, 113, 0, 25, 0);
insert into products values (62, 'tarte au sucre', 29, 3, '48 pies', 49.2999992, 17, 0, 0, 0);
insert into products values (63, 'vegie-spread', 7, 2, '15 - 625 g jars', 43.9000015, 24, 0, 5, 0);
insert into products values (64, 'wimmers gute semmelknödel', 12, 5, '20 bags x 4 pieces', 33.25, 22, 80, 30, 0);
insert into products values (65, 'louisiana fiery hot pepper sauce', 2, 2, '32 - 8 oz bottles', 21.0499992, 76, 0, 0, 0);
insert into products values (66, 'louisiana hot spiced okra', 2, 2, '24 - 8 oz jars', 17, 4, 100, 20, 0);
insert into products values (67, 'laughing lumberjack lager', 16, 1, '24 - 12 oz bottles', 14, 52, 0, 10, 0);
insert into products values (68, 'scottish longbreads', 8, 3, '10 boxes x 8 pieces', 12.5, 6, 10, 15, 0);
insert into products values (69, 'gudbrandsdalsost', 15, 4, '10 kg pkg.', 36, 26, 0, 15, 0);
insert into products values (70, 'outback lager', 7, 1, '24 - 355 ml bottles', 15, 15, 10, 30, 0);
insert into products values (71, 'flotemysost', 15, 4, '10 - 500 g pkgs.', 21.5, 26, 0, 0, 0);
insert into products values (72, 'mozzarella di giovanni', 14, 4, '24 - 200 g pkgs.', 34.7999992, 14, 0, 0, 0);
insert into products values (73, 'röd kaviar', 17, 8, '24 - 150 g jars', 15, 101, 0, 5, 0);
insert into products values (74, 'longlife tofu', 4, 7, '5 kg pkg.', 10, 4, 20, 5, 0);
insert into products values (75, 'rhönbräu klosterbier', 12, 1, '24 - 0.5 l bottles', 7.75, 125, 0, 25, 0);
insert into products values (76, 'lakkalikööri', 23, 1, '500 ml', 18, 57, 0, 20, 0);
insert into products values (77, 'original frankfurter grüne soße', 12, 2, '12 boxes', 13, 32, 0, 15, 0);


--
-- data for name: region; type: table data; schema: public; owner: postgres
--

insert into region values (1, 'eastern');
insert into region values (2, 'western');
insert into region values (3, 'northern');
insert into region values (4, 'southern');


--
-- data for name: shippers; type: table data; schema: public; owner: postgres
--

insert into shippers values (1, 'speedy express', '(503) 555-9831');
insert into shippers values (2, 'united package', '(503) 555-3199');
insert into shippers values (3, 'federal shipping', '(503) 555-9931');
insert into shippers values (4, 'alliance shippers', '1-800-222-0451');
insert into shippers values (5, 'ups', '1-800-782-7892');
insert into shippers values (6, 'dhl', '1-800-225-5345');


--
-- data for name: shippers_tmp; type: table data; schema: public; owner: postgres
--

insert into shippers_tmp values (1, 'speedy express', '(503) 555-9831');
insert into shippers_tmp values (2, 'united package', '(503) 555-3199');
insert into shippers_tmp values (3, 'federal shipping', '(503) 555-9931');
insert into shippers_tmp values (4, 'alliance shippers', '1-800-222-0451');
insert into shippers_tmp values (5, 'ups', '1-800-782-7892');
insert into shippers_tmp values (6, 'dhl', '1-800-225-5345');


--
-- data for name: suppliers; type: table data; schema: public; owner: postgres
--

insert into suppliers values (1, 'exotic liquids', 'charlotte cooper', 'purchasing manager', '49 gilbert st.', 'london', null, 'ec1 4sd', 'uk', '(171) 555-2222', null, null);
insert into suppliers values (2, 'new orleans cajun delights', 'shelley burke', 'order administrator', 'p.o. box 78934', 'new orleans', 'la', '70117', 'usa', '(100) 555-4822', null, '#cajun.htm#');
insert into suppliers values (3, 'grandma kelly''s homestead', 'regina murphy', 'sales representative', '707 oxford rd.', 'ann arbor', 'mi', '48104', 'usa', '(313) 555-5735', '(313) 555-3349', null);
insert into suppliers values (4, 'tokyo traders', 'yoshi nagase', 'marketing manager', '9-8 sekimai musashino-shi', 'tokyo', null, '100', 'japan', '(03) 3555-5011', null, null);
insert into suppliers values (5, 'cooperativa de quesos ''las cabras''', 'antonio del valle saavedra', 'export administrator', 'calle del rosal 4', 'oviedo', 'asturias', '33007', 'spain', '(98) 598 76 54', null, null);
insert into suppliers values (6, 'mayumi''s', 'mayumi ohno', 'marketing representative', '92 setsuko chuo-ku', 'osaka', null, '545', 'japan', '(06) 431-7877', null, 'mayumi''s (on the world wide web)#http://www.microsoft.com/accessdev/sampleapps/mayumi.htm#');
insert into suppliers values (7, 'pavlova, ltd.', 'ian devling', 'marketing manager', '74 rose st. moonie ponds', 'melbourne', 'victoria', '3058', 'australia', '(03) 444-2343', '(03) 444-6588', null);
insert into suppliers values (8, 'specialty biscuits, ltd.', 'peter wilson', 'sales representative', '29 king''s way', 'manchester', null, 'm14 gsd', 'uk', '(161) 555-4448', null, null);
insert into suppliers values (9, 'pb knäckebröd ab', 'lars peterson', 'sales agent', 'kaloadagatan 13', 'göteborg', null, 's-345 67', 'sweden', '031-987 65 43', '031-987 65 91', null);
insert into suppliers values (10, 'refrescos americanas ltda', 'carlos diaz', 'marketing manager', 'av. das americanas 12.890', 'sao paulo', null, '5442', 'brazil', '(11) 555 4640', null, null);
insert into suppliers values (11, 'heli süßwaren gmbh & co. kg', 'petra winkler', 'sales manager', 'tiergartenstraße 5', 'berlin', null, '10785', 'germany', '(010) 9984510', null, null);
insert into suppliers values (12, 'plutzer lebensmittelgroßmärkte ag', 'martin bein', 'international marketing mgr.', 'bogenallee 51', 'frankfurt', null, '60439', 'germany', '(069) 992755', null, 'plutzer (on the world wide web)#http://www.microsoft.com/accessdev/sampleapps/plutzer.htm#');
insert into suppliers values (13, 'nord-ost-fisch handelsgesellschaft mbh', 'sven petersen', 'coordinator foreign markets', 'frahmredder 112a', 'cuxhaven', null, '27478', 'germany', '(04721) 8713', '(04721) 8714', null);
insert into suppliers values (14, 'formaggi fortini s.r.l.', 'elio rossi', 'sales representative', 'viale dante, 75', 'ravenna', null, '48100', 'italy', '(0544) 60323', '(0544) 60603', '#formaggi.htm#');
insert into suppliers values (15, 'norske meierier', 'beate vileid', 'marketing manager', 'hatlevegen 5', 'sandvika', null, '1320', 'norway', '(0)2-953010', null, null);
insert into suppliers values (16, 'bigfoot breweries', 'cheryl saylor', 'regional account rep.', '3400 - 8th avenue suite 210', 'bend', 'or', '97101', 'usa', '(503) 555-9931', null, null);
insert into suppliers values (17, 'svensk sjöföda ab', 'michael björn', 'sales representative', 'brovallavägen 231', 'stockholm', null, 's-123 45', 'sweden', '08-123 45 67', null, null);
insert into suppliers values (18, 'aux joyeux ecclésiastiques', 'guylène nodier', 'sales manager', '203, rue des francs-bourgeois', 'paris', null, '75004', 'france', '(1) 03.83.00.68', '(1) 03.83.00.62', null);
insert into suppliers values (19, 'new england seafood cannery', 'robb merchant', 'wholesale account agent', 'order processing dept. 2100 paul revere blvd.', 'boston', 'ma', '02134', 'usa', '(617) 555-3267', '(617) 555-3389', null);
insert into suppliers values (20, 'leka trading', 'chandra leka', 'owner', '471 serangoon loop, suite #402', 'singapore', null, '0512', 'singapore', '555-8787', null, null);
insert into suppliers values (21, 'lyngbysild', 'niels petersen', 'sales manager', 'lyngbysild fiskebakken 10', 'lyngby', null, '2800', 'denmark', '43844108', '43844115', null);
insert into suppliers values (22, 'zaanse snoepfabriek', 'dirk luchte', 'accounting manager', 'verkoop rijnweg 22', 'zaandam', null, '9999 zz', 'netherlands', '(12345) 1212', '(12345) 1210', null);
insert into suppliers values (23, 'karkki oy', 'anne heikkonen', 'product manager', 'valtakatu 12', 'lappeenranta', null, '53120', 'finland', '(953) 10956', null, null);
insert into suppliers values (24, 'g''day, mate', 'wendy mackenzie', 'sales representative', '170 prince edward parade hunter''s hill', 'sydney', 'nsw', '2042', 'australia', '(02) 555-5914', '(02) 555-4873', 'g''day mate (on the world wide web)#http://www.microsoft.com/accessdev/sampleapps/gdaymate.htm#');
insert into suppliers values (25, 'ma maison', 'jean-guy lauzon', 'marketing manager', '2960 rue st. laurent', 'montréal', 'québec', 'h1j 1c3', 'canada', '(514) 555-9022', null, null);
insert into suppliers values (26, 'pasta buttini s.r.l.', 'giovanni giudici', 'order administrator', 'via dei gelsomini, 153', 'salerno', null, '84100', 'italy', '(089) 6547665', '(089) 6547667', null);
insert into suppliers values (27, 'escargots nouveaux', 'marie delamare', 'sales manager', '22, rue h. voiron', 'montceau', null, '71300', 'france', '85.57.00.07', null, null);
insert into suppliers values (28, 'gai pâturage', 'eliane noz', 'sales representative', 'bat. b 3, rue des alpes', 'annecy', null, '74000', 'france', '38.76.98.06', '38.76.98.58', null);
insert into suppliers values (29, 'forêts d''érables', 'chantal goulet', 'accounting manager', '148 rue chasseur', 'ste-hyacinthe', 'québec', 'j2s 7s8', 'canada', '(514) 555-2955', '(514) 555-2921', null);


--
-- data for name: territories; type: table data; schema: public; owner: postgres
--

insert into territories values ('01581', 'westboro', 1);
insert into territories values ('01730', 'bedford', 1);
insert into territories values ('01833', 'georgetow', 1);
insert into territories values ('02116', 'boston', 1);
insert into territories values ('02139', 'cambridge', 1);
insert into territories values ('02184', 'braintree', 1);
insert into territories values ('02903', 'providence', 1);
insert into territories values ('03049', 'hollis', 3);
insert into territories values ('03801', 'portsmouth', 3);
insert into territories values ('06897', 'wilton', 1);
insert into territories values ('07960', 'morristown', 1);
insert into territories values ('08837', 'edison', 1);
insert into territories values ('10019', 'new york', 1);
insert into territories values ('10038', 'new york', 1);
insert into territories values ('11747', 'mellvile', 1);
insert into territories values ('14450', 'fairport', 1);
insert into territories values ('19428', 'philadelphia', 3);
insert into territories values ('19713', 'neward', 1);
insert into territories values ('20852', 'rockville', 1);
insert into territories values ('27403', 'greensboro', 1);
insert into territories values ('27511', 'cary', 1);
insert into territories values ('29202', 'columbia', 4);
insert into territories values ('30346', 'atlanta', 4);
insert into territories values ('31406', 'savannah', 4);
insert into territories values ('32859', 'orlando', 4);
insert into territories values ('33607', 'tampa', 4);
insert into territories values ('40222', 'louisville', 1);
insert into territories values ('44122', 'beachwood', 3);
insert into territories values ('45839', 'findlay', 3);
insert into territories values ('48075', 'southfield', 3);
insert into territories values ('48084', 'troy', 3);
insert into territories values ('48304', 'bloomfield hills', 3);
insert into territories values ('53404', 'racine', 3);
insert into territories values ('55113', 'roseville', 3);
insert into territories values ('55439', 'minneapolis', 3);
insert into territories values ('60179', 'hoffman estates', 2);
insert into territories values ('60601', 'chicago', 2);
insert into territories values ('72716', 'bentonville', 4);
insert into territories values ('75234', 'dallas', 4);
insert into territories values ('78759', 'austin', 4);
insert into territories values ('80202', 'denver', 2);
insert into territories values ('80909', 'colorado springs', 2);
insert into territories values ('85014', 'phoenix', 2);
insert into territories values ('85251', 'scottsdale', 2);
insert into territories values ('90405', 'santa monica', 2);
insert into territories values ('94025', 'menlo park', 2);
insert into territories values ('94105', 'san francisco', 2);
insert into territories values ('95008', 'campbell', 2);
insert into territories values ('95054', 'santa clara', 2);
insert into territories values ('95060', 'santa cruz', 2);
insert into territories values ('98004', 'bellevue', 2);
insert into territories values ('98052', 'redmond', 2);
insert into territories values ('98104', 'seattle', 2);


--
-- data for name: usstates; type: table data; schema: public; owner: postgres
--

insert into usstates values (1, 'alabama', 'al', 'south');
insert into usstates values (2, 'alaska', 'ak', 'north');
insert into usstates values (3, 'arizona', 'az', 'west');
insert into usstates values (4, 'arkansas', 'ar', 'south');
insert into usstates values (5, 'california', 'ca', 'west');
insert into usstates values (6, 'colorado', 'co', 'west');
insert into usstates values (7, 'connecticut', 'ct', 'east');
insert into usstates values (8, 'delaware', 'de', 'east');
insert into usstates values (9, 'district of columbia', 'dc', 'east');
insert into usstates values (10, 'florida', 'fl', 'south');
insert into usstates values (11, 'georgia', 'ga', 'south');
insert into usstates values (12, 'hawaii', 'hi', 'west');
insert into usstates values (13, 'idaho', 'id', 'midwest');
insert into usstates values (14, 'illinois', 'il', 'midwest');
insert into usstates values (15, 'indiana', 'in', 'midwest');
insert into usstates values (16, 'iowa', 'io', 'midwest');
insert into usstates values (17, 'kansas', 'ks', 'midwest');
insert into usstates values (18, 'kentucky', 'ky', 'south');
insert into usstates values (19, 'louisiana', 'la', 'south');
insert into usstates values (20, 'maine', 'me', 'north');
insert into usstates values (21, 'maryland', 'md', 'east');
insert into usstates values (22, 'massachusetts', 'ma', 'north');
insert into usstates values (23, 'michigan', 'mi', 'north');
insert into usstates values (24, 'minnesota', 'mn', 'north');
insert into usstates values (25, 'mississippi', 'ms', 'south');
insert into usstates values (26, 'missouri', 'mo', 'south');
insert into usstates values (27, 'montana', 'mt', 'west');
insert into usstates values (28, 'nebraska', 'ne', 'midwest');
insert into usstates values (29, 'nevada', 'nv', 'west');
insert into usstates values (30, 'new hampshire', 'nh', 'east');
insert into usstates values (31, 'new jersey', 'nj', 'east');
insert into usstates values (32, 'new mexico', 'nm', 'west');
insert into usstates values (33, 'new york', 'ny', 'east');
insert into usstates values (34, 'north carolina', 'nc', 'east');
insert into usstates values (35, 'north dakota', 'nd', 'midwest');
insert into usstates values (36, 'ohio', 'oh', 'midwest');
insert into usstates values (37, 'oklahoma', 'ok', 'midwest');
insert into usstates values (38, 'oregon', 'or', 'west');
insert into usstates values (39, 'pennsylvania', 'pa', 'east');
insert into usstates values (40, 'rhode island', 'ri', 'east');
insert into usstates values (41, 'south carolina', 'sc', 'east');
insert into usstates values (42, 'south dakota', 'sd', 'midwest');
insert into usstates values (43, 'tennessee', 'tn', 'midwest');
insert into usstates values (44, 'texas', 'tx', 'west');
insert into usstates values (45, 'utah', 'ut', 'west');
insert into usstates values (46, 'vermont', 'vt', 'east');
insert into usstates values (47, 'virginia', 'va', 'east');
insert into usstates values (48, 'washington', 'wa', 'west');
insert into usstates values (49, 'west virginia', 'wv', 'south');
insert into usstates values (50, 'wisconsin', 'wi', 'midwest');
insert into usstates values (51, 'wyoming', 'wy', 'west');


--
-- name: pk_categories; type: constraint; schema: public; owner: postgres; tablespace: 
--

alter table only categories
    add constraint pk_categories primary key ("categoryid");


--
-- name: pk_customercustomerdemo; type: constraint; schema: public; owner: postgres; tablespace: 
--

alter table only customercustomerdemo
    add constraint pk_customercustomerdemo primary key ("customerid", "customertypeid");


--
-- name: pk_customerdemographics; type: constraint; schema: public; owner: postgres; tablespace: 
--

alter table only customerdemographics
    add constraint pk_customerdemographics primary key ("customertypeid");


--
-- name: pk_customers; type: constraint; schema: public; owner: postgres; tablespace: 
--

alter table only customers
    add constraint pk_customers primary key ("customerid");


--
-- name: pk_employees; type: constraint; schema: public; owner: postgres; tablespace: 
--

alter table only employees
    add constraint pk_employees primary key ("employeeid");


--
-- name: pk_employeeterritories; type: constraint; schema: public; owner: postgres; tablespace: 
--

alter table only employeeterritories
    add constraint pk_employeeterritories primary key ("employeeid", "territoryid");


--
-- name: pk_order_details; type: constraint; schema: public; owner: postgres; tablespace: 
--

alter table only order_details
    add constraint pk_order_details primary key ("orderid", "productid");


--
-- name: pk_orders; type: constraint; schema: public; owner: postgres; tablespace: 
--

alter table only orders
    add constraint pk_orders primary key ("orderid");


--
-- name: pk_products; type: constraint; schema: public; owner: postgres; tablespace: 
--

alter table only products
    add constraint pk_products primary key ("productid");


--
-- name: pk_region; type: constraint; schema: public; owner: postgres; tablespace: 
--

alter table only region
    add constraint pk_region primary key ("regionid");


--
-- name: pk_shippers; type: constraint; schema: public; owner: postgres; tablespace: 
--

alter table only shippers
    add constraint pk_shippers primary key ("shipperid");


--
-- name: pk_shippers_tmp; type: constraint; schema: public; owner: postgres; tablespace: 
--

alter table only shippers_tmp
    add constraint pk_shippers_tmp primary key ("shipperid");


--
-- name: pk_suppliers; type: constraint; schema: public; owner: postgres; tablespace: 
--

alter table only suppliers
    add constraint pk_suppliers primary key ("supplierid");


--
-- name: pk_territories; type: constraint; schema: public; owner: postgres; tablespace: 
--

alter table only territories
    add constraint pk_territories primary key ("territoryid");


--
-- name: public; type: acl; schema: -; owner: postgres
--

revoke all on schema public from public;
revoke all on schema public from postgres;
grant all on schema public to postgres;
grant all on schema public to public;


--
-- postgresql database dump complete
--

