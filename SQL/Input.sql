With Expectation AS (
		SELECT
			b.stars as BusinessAverageStars,
			cast(ROUND(u._average_stars_ * 4, 0) / 4 as decimal(10, 2)) as UserAverageStars,
			AVG(r._stars_ + 0.0) as ExpectedStars,
			ISNULL(VAR(r._stars_ + 0.0), 0) as VarStars
		FROM review r
			INNER JOIN [user] u
				ON u._user_id_ = r._user_id_
			INNER JOIN Business b
				ON b.business_id = r._business_id_
		GROUP BY b.stars, cast(ROUND(u._average_stars_ * 4, 0) / 4 as decimal(10, 2))),
	Optimisim AS (SELECT AVG(0.0 + r._stars_ - b.stars) Optimisim, ISNULL(Var(0.0 + r._stars_ - b.stars), 0.0) as OptimisimVar, r._user_id_ FROM review r
		INNER JOIN business b
			ON r._business_id_ = b.business_id
		INNER JOIN [user] u
			ON r._user_id_ = u._user_id_
		GROUP BY r._user_id_),
	FriendAverage AS (
		SELECT f.userId, AVG(u._average_stars_) FriendStars 
			FROM friends f INNER JOIN [user] u ON f.friendId = u._user_id_
		GROUP BY f.userId
			),
	Extreme AS(
		SELECT r._user_id_, SUM(CASE WHEN r._stars_ = 1 OR r._stars_ = 5 THEN 1.0 ELSE 0.0 END) AS ExtremeCount 
			FROM review r
			GROUP BY r._user_id_
	)
SELECT 
r._stars_ AS Stars,
u._average_stars_ AS UserAverage,
b.stars AS BusinessAverage,
e.ExpectedStars As ExpectedStars,
e.VarStars AS ExpectedVarStars,
u._review_count_ AS ReviewCount,
o.Optimisim AS Optimisim,
o.OptimisimVar as OptimisimVar,
ISNULL(f.FriendStars - u._average_stars_, 0.0) AS FriendsAverageStars,
(extr.ExtremeCount * 1.0) / u._review_count_ AS Extremeness





FROM review r
INNER JOIN business b
	ON r._business_id_ = b.business_id
INNER JOIN [user] u
	ON r._user_id_ = u._user_id_
INNER JOIN Optimisim o
	ON r._user_id_ = o._user_id_
LEFT JOIN FriendAverage f
	ON f.userId = r._user_id_
INNER JOIN Expectation e
	ON cast(ROUND(u._average_stars_ * 4, 0) / 4 as decimal(10, 2)) = e.UserAverageStars AND
		b.stars = e.BusinessAverageStars
INNER JOIN Extreme extr
	ON u._user_id_ = extr._user_id_

WHERE u._review_count_ <> 0