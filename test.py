text = """SUM( SELECT( CountyMonthVoucherLineItem[AmountApproved], IN( [CountyMonthVoucherID],SELECT(CountyMonthVoucher[ID], AND( [MonthID] < [_THISROW].[MonthID], [CountyFiscalYearBudgetID] = [_THISROW].[CountyFiscalYearBudgetID] ) ) ) ) )"""

print(text.replace('\n',' ').replace('  ',''))

