create table observation (
	obs_id serial primary key,
	state bytea,
	action int not null,
	reward float not null,
	done boolean not null,
	step int not null,
	episode int not null
);
