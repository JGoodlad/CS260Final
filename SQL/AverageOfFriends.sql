SET NOCOUNT ON

--CREATE TABLE friends(userId nvarchar(50) , friendId nvarchar(50))--, friendavg float)

DECLARE @Name VARCHAR(100)
DECLARE @Friends NVARCHAR(MAX)
DECLARE Split_friends CURSOR LOCAL FAST_FORWARD FOR 
        SELECT _user_id_ , _friends_
        FROM [user]

OPEN Split_friends

FETCH NEXT FROM Split_friends 
INTO @Name, @Friends
 
WHILE @@FETCH_STATUS = 0
BEGIN

INSERT INTO friends
        SELECT @Name, trim(SPL.value)--, uu._average_stars_
        FROM STRING_SPLIT(@Friends,',') AS SPL
		--INNER JOIN [user] uu
		--	ON TRIM(SPL.value) = uu._user_id_

		FETCH NEXT FROM Split_friends 
        INTO @Name, @Friends

End

SELECT * FROM friends

--DROP TABLE #TempFriends