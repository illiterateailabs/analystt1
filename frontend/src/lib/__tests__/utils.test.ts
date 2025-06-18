import { formatAddress, formatAmount, formatUSD, cn, getInitials } from '../utils';

describe('Utility Functions', () => {
  describe('formatAddress', () => {
    test('should format a standard Ethereum address', () => {
      const address = '0x1234567890abcdef1234567890abcdef12345678';
      expect(formatAddress(address)).toBe('0x1234...5678');
    });

    test('should format a short address', () => {
      const address = '0xabc';
      expect(formatAddress(address)).toBe('0xabc');
    });

    test('should handle an address shorter than the desired format', () => {
      const address = '0x1234567890';
      expect(formatAddress(address)).toBe('0x1234567890');
    });

    test('should return empty string for null or undefined address', () => {
      expect(formatAddress(null)).toBe('');
      expect(formatAddress(undefined)).toBe('');
    });

    test('should return empty string for an empty address', () => {
      expect(formatAddress('')).toBe('');
    });

    test('should handle non-string input gracefully', () => {
      expect(formatAddress(12345 as any)).toBe('');
      expect(formatAddress({} as any)).toBe('');
    });
  });

  describe('formatAmount', () => {
    test('should format a large amount with 18 decimals', () => {
      const amount = '1000000000000000000000'; // 1000 ETH
      expect(formatAmount(amount, 18)).toBe('1,000.00');
    });

    test('should format a small amount with 18 decimals', () => {
      const amount = '123456789012345678'; // 0.123456789012345678 ETH
      expect(formatAmount(amount, 18)).toBe('0.12');
    });

    test('should format an amount with 6 decimals', () => {
      const amount = '1234567'; // 1.234567 USDC
      expect(formatAmount(amount, 6)).toBe('1.23');
    });

    test('should format an amount with 0 decimals', () => {
      const amount = '500';
      expect(formatAmount(amount, 0)).toBe('500');
    });

    test('should handle zero amount', () => {
      const amount = '0';
      expect(formatAmount(amount, 18)).toBe('0.00');
    });

    test('should handle negative amount', () => {
      const amount = '-1000000000000000000';
      expect(formatAmount(amount, 18)).toBe('-1,000.00');
    });

    test('should handle amount with fewer digits than decimals', () => {
      const amount = '123';
      expect(formatAmount(amount, 5)).toBe('0.00'); // 0.00123, rounded to 2 decimal places
    });

    test('should handle invalid amount string', () => {
      const amount = 'not-a-number';
      expect(formatAmount(amount, 18)).toBe('0.00');
    });

    test('should handle null or undefined amount', () => {
      expect(formatAmount(null, 18)).toBe('0.00');
      expect(formatAmount(undefined, 18)).toBe('0.00');
    });

    test('should handle null or undefined decimals', () => {
      const amount = '1000000000000000000';
      expect(formatAmount(amount, null)).toBe('1.00'); // Defaults to 18 decimals
      expect(formatAmount(amount, undefined)).toBe('1.00'); // Defaults to 18 decimals
    });

    test('should abbreviate large numbers', () => {
      const amount = '1000000000000000000000000'; // 1,000,000 ETH
      expect(formatAmount(amount, 18)).toBe('1.00M');
    });

    test('should abbreviate very large numbers', () => {
      const amount = '1000000000000000000000000000'; // 1,000,000,000 ETH
      expect(formatAmount(amount, 18)).toBe('1.00B');
    });

    test('should abbreviate numbers with K suffix', () => {
      const amount = '123456789012345678901'; // 123.45 ETH
      expect(formatAmount(amount, 18)).toBe('123.46');
      const amountK = '12345678901234567890123'; // 12345.67 ETH
      expect(formatAmount(amountK, 18)).toBe('12.35K');
    });
  });

  describe('formatUSD', () => {
    test('should format a positive USD value', () => {
      expect(formatUSD(1234.56)).toBe('$1,234.56');
    });

    test('should format a zero USD value', () => {
      expect(formatUSD(0)).toBe('$0.00');
    });

    test('should format a negative USD value', () => {
      expect(formatUSD(-123.45)).toBe('-$123.45');
    });

    test('should format a large USD value', () => {
      expect(formatUSD(1234567.89)).toBe('$1,234,567.89');
    });

    test('should handle null or undefined input', () => {
      expect(formatUSD(null)).toBe('$0.00');
      expect(formatUSD(undefined)).toBe('$0.00');
    });

    test('should handle non-numeric input', () => {
      expect(formatUSD('abc' as any)).toBe('$0.00');
    });

    test('should round to two decimal places', () => {
      expect(formatUSD(123.456)).toBe('$123.46');
      expect(formatUSD(123.454)).toBe('$123.45');
    });
  });

  describe('cn', () => {
    test('should merge class names correctly', () => {
      expect(cn('class1', 'class2')).toBe('class1 class2');
    });

    test('should handle conditional classes', () => {
      expect(cn('class1', true && 'class2', false && 'class3')).toBe('class1 class2');
    });

    test('should filter out falsy values', () => {
      expect(cn('class1', null, undefined, '', 0, false, 'class2')).toBe('class1 class2');
    });

    test('should handle arrays of class names', () => {
      expect(cn(['class1', 'class2'], 'class3')).toBe('class1 class2 class3');
    });

    test('should handle nested arrays', () => {
      expect(cn(['class1', ['class2', 'class3']], 'class4')).toBe('class1 class2 class3 class4');
    });

    test('should handle objects for conditional classes', () => {
      expect(cn({ class1: true, class2: false }, 'class3')).toBe('class1 class3');
    });

    test('should handle mixed input types', () => {
      expect(cn('class1', ['class2', { class3: true, class4: false }], 'class5')).toBe('class1 class2 class3 class5');
    });

    test('should return empty string if no valid classes are provided', () => {
      expect(cn(null, undefined, false, '')).toBe('');
    });
  });

  describe('getInitials', () => {
    test('should return initials for a full name', () => {
      expect(getInitials('John Doe')).toBe('JD');
    });

    test('should return initials for a single name', () => {
      expect(getInitials('John')).toBe('J');
    });

    test('should handle names with multiple spaces', () => {
      expect(getInitials('  John   Doe  ')).toBe('JD');
    });

    test('should handle names with more than two parts', () => {
      expect(getInitials('John David Doe')).toBe('JD');
    });

    test('should handle empty string', () => {
      expect(getInitials('')).toBe('');
    });

    test('should handle null or undefined input', () => {
      expect(getInitials(null)).toBe('');
      expect(getInitials(undefined)).toBe('');
    });

    test('should handle non-string input', () => {
      expect(getInitials(123 as any)).toBe('');
      expect(getInitials({} as any)).toBe('');
    });

    test('should handle names with leading/trailing spaces', () => {
      expect(getInitials(' Jane ')).toBe('J');
      expect(getInitials('  Alice Wonderland  ')).toBe('AW');
    });
  });
});
