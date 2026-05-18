USE banking_fraud

SELECT COUNT(*) AS total_fraud
FROM credit_card
WHERE Class = 1;


SELECT COUNT(*) AS genuine_transactions
FROM credit_card
WHERE Class = 0;


SELECT 
    (SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) 
    AS fraud_percentage
FROM credit_card;


SELECT MAX(Amount) AS highest_fraud_amount
FROM credit_card
WHERE Class = 1;


SELECT AVG(Amount) AS avg_fraud_amount
FROM credit_card
WHERE Class = 1;


SELECT 
    Class,
    AVG(Amount) AS avg_amount
FROM credit_card
GROUP BY Class;


SELECT *
FROM credit_card
WHERE Class = 1
ORDER BY Amount DESC
LIMIT 10;


SELECT *
FROM credit_card
WHERE Amount > 10000;


SELECT *
FROM credit_card
WHERE Class = 1
AND Amount > (
    SELECT AVG(Amount)
    FROM credit_card
);



SELECT Time, COUNT(*) AS fraud_count
FROM credit_card
WHERE Class = 1
GROUP BY Time
ORDER BY fraud_count DESC;



SELECT V1, Amount
FROM credit_card
WHERE Class = 1
ORDER BY V1 ASC
LIMIT 10;



SELECT *
FROM credit_card
WHERE Class = 1
AND Amount > 500;



SELECT SUM(Amount) AS total_fraud_loss
FROM credit_card
WHERE Class = 1;



SELECT 
    FLOOR(Time / 3600) AS hour,
    COUNT(*) AS fraud_cases
FROM credit_card
WHERE Class = 1
GROUP BY hour
ORDER BY hour;



SELECT 
    Class,
    COUNT(*) AS total
FROM credit_card
GROUP BY Class;